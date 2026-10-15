"""Cartesian reaching task for the iiwa7 demonstrations (Gym 0.15 API)."""

import json
import os
from pathlib import Path

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "icrl/expert_data/KukaCup1/task_config.json"

class KukaCup1(gym.Env):
    """Reach a goal with a point end effector and a vertical cylinder cost.

    The cylinder is a learned task constraint, not a collision response: actions
    may enter it so that PUCL can collect infeasible samples. Cost ignores z and
    action. Observations are [position, goal - position, last displacement].
    Successful episode return is exactly minus the accumulated path length.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config_path=None, obstacle_center=None, obstacle_radius=None,
                 goal=None, start=None, workspace_low=None, workspace_high=None,
                 step_size=None, goal_tolerance=None, max_steps=None):
        path = Path(config_path or os.environ.get('KUKACUP1_CONFIG') or DEFAULT_CONFIG)
        config = json.loads(path.read_text()) if path.exists() else {}
        self.obstacle_center = np.asarray(
            obstacle_center if obstacle_center is not None else config.get("obstacle_center"),
            dtype=np.float64,
        )
        radius = obstacle_radius if obstacle_radius is not None else config.get("obstacle_radius")
        if self.obstacle_center.shape != (2,) or not np.isfinite(self.obstacle_center).all() or radius is None:
            raise ValueError("Set KukaCup1 obstacle_center [x, y] and obstacle_radius in "
                             f"{path}, or pass them to gym.make(). Coordinates are metres in iiwa7_link_0.")
        self.obstacle_radius = float(radius)
        if not np.isfinite(self.obstacle_radius) or self.obstacle_radius <= 0:
            raise ValueError("obstacle_radius must be finite and positive")
        self.workspace_low = np.asarray(workspace_low if workspace_low is not None else
                                        config.get("workspace_low", [0.3, -0.4, 0.1]), dtype=np.float64)
        self.workspace_high = np.asarray(workspace_high if workspace_high is not None else
                                         config.get("workspace_high", [0.9, 0.3, 0.6]), dtype=np.float64)
        if (self.workspace_low.shape != (3,) or self.workspace_high.shape != (3,) or
                not np.isfinite([self.workspace_low, self.workspace_high]).all() or
                np.any(self.workspace_high <= self.workspace_low)):
            raise ValueError("workspace bounds must be finite ordered 3-vectors")
        self.step_size = float(step_size if step_size is not None else config.get("step_size", 0.008))
        self.goal_tolerance = float(goal_tolerance if goal_tolerance is not None else
                                    config.get("goal_tolerance", 0.01))
        self.max_steps = int(max_steps if max_steps is not None else config.get("max_steps", 200))
        if (not np.isfinite([self.step_size, self.goal_tolerance]).all() or
                self.step_size <= 0 or self.goal_tolerance <= 0 or self.max_steps <= 0):
            raise ValueError("step_size, goal_tolerance and max_steps must be positive")
        if self.max_steps > 200:
            raise ValueError("max_steps must not exceed the registered 200-step time limit")
        tasks = config.get("tasks", [])
        fallback = {"start": [0.71, -0.22, 0.26], "goal": [0.45, 0.067, 0.269]}
        default_task = tasks[0] if tasks else fallback
        self.default_start = self._position(start if start is not None else default_task["start"])
        self.default_goal = self._position(goal if goal is not None else default_task["goal"])
        self.default_tasks = [(self._position(task["start"]), self._position(task["goal"]))
                              for task in tasks] if start is None and goal is None else []
        self.starting_point_candidates = None
        self.ending_point_candidates = None
        self.task_config_idx = 0
        self.action_space = spaces.Box(-np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32))
        span = self.workspace_high - self.workspace_low
        self.observation_space = spaces.Box(
            np.concatenate([self.workspace_low, -span, -np.ones(3) * self.step_size]).astype(np.float32),
            np.concatenate([self.workspace_high, span, np.ones(3) * self.step_size]).astype(np.float32),
        )
        # A stationary policy must not beat a successful reaching trajectory.
        # Failure incurs more than the maximum possible episode path length.
        self.failure_penalty = self.max_steps * self.step_size * np.sqrt(3.0) + 1.0
        self.current_state = None
        self.ending_point = self.default_goal.copy()
        self.task_config = {
            "coordinate_frame": config.get("coordinate_frame", "iiwa7_link_0"),
            "units": "metres",
            "obstacle_center": self.obstacle_center.tolist(),
            "obstacle_radius": self.obstacle_radius,
            "workspace_low": self.workspace_low.tolist(),
            "workspace_high": self.workspace_high.tolist(),
            "step_size": self.step_size,
            "goal_tolerance": self.goal_tolerance,
            "max_steps": self.max_steps,
            "tasks": [{"start": s.tolist(), "goal": g.tolist()} for s, g in self.default_tasks],
        }
        self.seed()

    def _position(self, value):
        position = np.asarray(value, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("positions must be finite [x, y, z] vectors")
        if np.any(position < self.workspace_low) or np.any(position > self.workspace_high):
            raise ValueError("position lies outside the workspace")
        return position.copy()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_starting_point(self, starting_point, ending_point=None):
        if starting_point is None:
            self.starting_point_candidates = self.ending_point_candidates = None
        else:
            values = np.atleast_2d(np.asarray(starting_point, dtype=np.float64))
            if values.shape[1] not in (3, 9) or not np.isfinite(values).all():
                raise ValueError("starting points must have shape (N, 3) or (N, 9)")
            self.starting_point_candidates = np.array([self._position(v[:3]) for v in values])
            if ending_point is not None:
                goals = np.atleast_2d(np.asarray(ending_point, dtype=np.float64))
                if goals.shape != (len(values), 3):
                    raise ValueError("one ending point is required for each starting point")
            elif values.shape[1] == 9:
                goals = values[:, :3] + values[:, 3:6]
            else:
                goals = np.repeat(self.default_goal[None], len(values), axis=0)
            self.ending_point_candidates = np.array([self._position(v) for v in goals])
        self.task_config_idx = 0

    def reset(self, **kwargs):
        if self.starting_point_candidates is None:
            if self.default_tasks:
                start, goal = self.default_tasks[self.task_config_idx % len(self.default_tasks)]
                self.task_config_idx += 1
            else:
                start, goal = self.default_start, self.default_goal
            self.current_state = start.copy()
            self.ending_point = goal.copy()
        else:
            index = self.task_config_idx % len(self.starting_point_candidates)
            self.current_state = self.starting_point_candidates[index].copy()
            self.ending_point = self.ending_point_candidates[index].copy()
            self.task_config_idx += 1
        self.steps = 0
        self.path_length = 0.0
        self.cost_counter = int(self.true_cost_function(self.current_state, None))
        self.terminated = False
        self.observation = self._observation(np.zeros(3))
        return self.observation.copy()

    def _observation(self, displacement):
        return np.concatenate([self.current_state, self.ending_point - self.current_state,
                               displacement]).astype(np.float32)

    def step(self, action):
        if self.current_state is None or self.terminated:
            raise RuntimeError("call reset() before stepping a new episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite 3-vector")
        previous = self.current_state.copy()
        self.current_state = np.clip(previous + self.step_size * np.clip(action, -1, 1),
                                     self.workspace_low, self.workspace_high)
        displacement = self.current_state - previous
        distance = float(np.linalg.norm(displacement))
        self.path_length += distance
        self.steps += 1
        cost = float(self.true_cost_function(self.current_state, action))
        self.cost_counter += int(cost)
        reached = np.linalg.norm(self.current_state - self.ending_point) <= self.goal_tolerance
        timed_out = self.steps >= self.max_steps and not reached
        self.terminated = bool(reached or timed_out)
        reward = -distance - (self.failure_penalty if timed_out else 0.0)
        self.observation = self._observation(displacement)
        info = {"cost": cost, "is_success": bool(reached and self.cost_counter == 0),
                "goal_reached": bool(reached), "path_length": self.path_length,
                "dis_reward": -distance, "ctl_reward": 0.0,
                "failure_penalty": self.failure_penalty if timed_out else 0.0}
        if timed_out:
            info["TimeLimit.truncated"] = True
        return self.observation.copy(), float(reward), self.terminated, info

    def true_cost_function(self, obs, acs=None):
        positions = np.asarray(obs)[..., :2]
        return (np.sum((positions - self.obstacle_center) ** 2, axis=-1) <=
                self.obstacle_radius ** 2).astype(np.float32)

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise ValueError("only rgb_array rendering is supported")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle
        figure = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasAgg(figure)
        ax = figure.add_subplot(111)
        ax.add_patch(Circle(self.obstacle_center, self.obstacle_radius, color="red", alpha=0.4))
        ax.scatter(*self.ending_point[:2], marker="*", color="green", label="goal")
        if self.current_state is not None:
            ax.scatter(*self.current_state[:2], color="blue", label="end effector")
        ax.set(xlim=(self.workspace_low[0], self.workspace_high[0]),
               ylim=(self.workspace_low[1], self.workspace_high[1]), xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal")
        ax.legend()
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        pass
