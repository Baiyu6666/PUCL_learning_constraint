import math
import os
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

try:
    from gym.envs.mujoco import mujoco_env
except Exception:
    class _UnavailableMujocoEnv(gym.Env):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MuJoCo backend is unavailable. This env requires gym mujoco bindings."
            )

    class _MujocoModule:
        MujocoEnv = _UnavailableMujocoEnv

    mujoco_env = _MujocoModule()
# tets rmote
ABS_PATH = os.path.abspath(os.path.dirname(__file__))

# Constraint values
X_HIGH = +6
X_LOW  = -6

DEFAULT_CAMERA_CONFIG = {
    'distance': 15.0,
}

# point_with_null_reward and point_with_reward_on_track have been removed by Erfan.

# =========================================================================== #
#                           Point With Circle Reward                          #
# =========================================================================== #

class PointCircle(mujoco_env.MujocoEnv):
    def __init__(
            self,
            circle_reward=True,
            start_on_circle=True,
            xml_file=ABS_PATH+'/xmls/point_circle.xml',
            size=40,
            reward_dir=[0., 0.],
            target_radius=10.,
            reward_ctrl_weight=0.0,
            action_clip_value=0.25,
            mode=0,
            *args,
            **kwargs
        ):

        self.size = size
        self.start_on_circle = start_on_circle
        self.reward_dir = reward_dir
        self.target_radius = target_radius
        self.circle_reward = circle_reward
        self.reward_ctrl_weight = reward_ctrl_weight
        self.action_clip_value = action_clip_value
        self.mode = mode
        super(PointCircle, self).__init__(xml_file, 1)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flatten(),
            self.data.qvel.flat,
            self.get_body_com('torso').flat,
        ])

    def reset_model(self):  # used for constraint learning, policy learning, and testing
        qvel = self.init_qvel
        # self.init_qpos is always 0
        if self.mode == 0:
            # default: 5, Baiyu: 6
            x_scale = 5
            y_scale = 5
            qpos = self.init_qpos.copy()
            qpos[0] = qpos[0] + self.np_random.uniform(low=-x_scale, high=x_scale, size=1)
            qpos[1] = qpos[1] + self.np_random.uniform(low=-y_scale, high=y_scale, size=1)
            qpos[2] = self.np_random.uniform(low=-np.pi, high=np.pi)
        elif self.mode == 1:
            qpos = self.init_qpos + [0, 10, 0]
            qpos[2] = 0
            self.mode = 2
        elif self.mode == 2:
            qpos = self.init_qpos + [0, -10, 0]
            qpos[2] = -np.pi
            self.mode = 1
        elif self.mode == 3:
            x_init = 5
            y_init = 10
            qpos = self.init_qpos
            qpos[0] = 2*x_init*np.random.rand() - self.x_init
            qpos[1] = 2*y_init*np.random.rand() - self.y_init
            qpos[2] = self.np_random.uniform(low=-np.pi, high=np.pi)

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def new_step(self, action):
        self.do_simulation(action, self.frame_skip)
        qpos = np.copy()

    def step(self, action):
        action = np.clip(action, -self.action_clip_value, self.action_clip_value)
        action[0] = np.clip(action[0], -0.0, self.action_clip_value)   # it's helpful to avoid the mini circles.
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        # it's better to keep the orientation within -pi to pi for reducing the scope of state space.
        if qpos[2] > np.pi:
            qpos[2] = qpos[2] - 2 * np.pi
        elif qpos[2] < - np.pi:
            qpos[2] = qpos[2] + 2 * np.pi

        # computing the increment in each direction
        ori = qpos[2]
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # ensuring that the robot is within a reasonable range
        qpos[0] = np.clip(qpos[0]+dx, -self.size, self.size)
        qpos[1] = np.clip(qpos[1]+dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        next_obs = self._get_obs()
        reward_ctrl = np.sum(np.square(action))

        x, y = qpos[0], qpos[1]
        # reward = y * dx - x * dy
        # reward = action[0] if (y * dx - x * dy) > 0 else action[0]*0
        # reward = np.abs(action[0]) * np.sign(y * dx - x * dy)
        # delta = 0.5
        if np.sqrt(x**2 + y**2)>1e-2:
            reward = (y * dx - x * dy)/np.sqrt(x**2 + y**2 + 0e-5)
        else:
            reward = 0

        # reward = np.exp(-(np.abs(x)-self.target_radius)**2/2/delta**2)*dy - np.exp(-(np.abs(y)-self.target_radius)**2/2/delta**2)*dx
        reward /= (1 + np.abs(np.sqrt(x**2 + y**2) - self.target_radius))

        infos = {'circle_reward': reward,
                 'control_reward': reward_ctrl,
                 'action_1': action[0],
                 'action_2': action[1]}

        return next_obs, reward, False, infos

    def get_xy(self):
        qpos = self.data.qpos
        return qpos[0, 0], qpos[1, 0]

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class PointCircleTest(PointCircle):
    def step(self, action):
        action = np.clip(action, -self.action_clip_value, self.action_clip_value)
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]

        if qpos[2] > np.pi:
            ori = qpos[2] - 2 * np.pi
        elif qpos[2] < - np.pi:
            ori = qpos[2] + 2 * np.pi
        else:
            ori = qpos[2]

        # computing the increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # ensuring that the robot is within a reasonable range
        qpos[0] = np.clip(qpos[0]+dx, -self.size, self.size)
        qpos[1] = np.clip(qpos[1]+dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        next_obs = self._get_obs()
        reward_ctrl = np.sum(np.square(action))
        done = False

        x, y = qpos[0], qpos[1]
        if np.sqrt(x ** 2 + y ** 2) > 1e-2:
            reward = (y * dx - x * dy) / np.sqrt(x ** 2 + y ** 2 + 0e-5)
        else:
            reward = 0
        reward /= (1 + np.abs(np.sqrt(x**2 + y**2) - self.target_radius))
        if x > X_HIGH or x < X_LOW:
            reward = 0
            done = True
            # print('terminating in true environment')

        infos = {'circle_reward': reward,
                 'control_reward': reward_ctrl,
                 'action_1': action[0],
                 'action_2': action[1]}

        return next_obs, reward, done, infos


class PointHalfCircle(mujoco_env.MujocoEnv):
    def __init__(
            self,
            circle_reward=True,
            start_on_circle=True,
            xml_file=ABS_PATH+'/xmls/point_circle.xml',
            size=40,
            reward_dir=[0., 0.],
            target_radius=10.,
            reward_ctrl_weight=0.0,
            action_clip_value=0.25,
            mode=0,
            *args,
            **kwargs
        ):

        self.size = size
        self.start_on_circle = start_on_circle
        self.reward_dir = reward_dir
        self.target_radius = target_radius
        self.circle_reward = circle_reward
        self.reward_ctrl_weight = reward_ctrl_weight
        self.action_clip_value = action_clip_value
        self.mode = mode
        super(PointHalfCircle, self).__init__(xml_file, 1)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flatten(),
            self.data.qvel.flat,
            self.get_body_com('torso').flat,
        ])

    def reset_model(self):  # used for constraint learning, policy learning, and testing
        qvel = self.init_qvel
        # self.init_qpos is always 0
        if self.mode == 0:
            # default: 5, Baiyu: 6
            x_scale = 5
            y_scale = 5
            qpos = self.init_qpos.copy()
            qpos[0] = qpos[0] + self.np_random.uniform(low=-x_scale, high=x_scale, size=1)
            qpos[1] = qpos[1] + self.np_random.uniform(low=-y_scale, high=y_scale, size=1)
            qpos[2] = self.np_random.uniform(low=-np.pi, high=np.pi)
        elif self.mode == 1:
            qpos = self.init_qpos + [0, 10, 0]
            qpos[2] = 0
            self.mode = 2
        elif self.mode == 2:
            qpos = self.init_qpos + [0, -10, 0]
            qpos[2] = -np.pi
            self.mode = 1
        elif self.mode == 3:
            x_init = 5
            y_init = 10
            qpos = self.init_qpos
            qpos[0] = 2*x_init*np.random.rand() - self.x_init
            qpos[1] = 2*y_init*np.random.rand() - self.y_init
            qpos[2] = self.np_random.uniform(low=-np.pi, high=np.pi)

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def new_step(self, action):
        self.do_simulation(action, self.frame_skip)
        qpos = np.copy()

    def step(self, action):
        action = np.clip(action, -self.action_clip_value, self.action_clip_value)
        action[0] = np.clip(action[0], -0.0, self.action_clip_value)   # it's helpful to avoid the mini circles.
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        # it's better to keep the orientation within -pi to pi for reducing the scope of state space.
        if qpos[2] > np.pi:
            qpos[2] = qpos[2] - 2 * np.pi
        elif qpos[2] < - np.pi:
            qpos[2] = qpos[2] + 2 * np.pi

        # computing the increment in each direction
        ori = qpos[2]
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # ensuring that the robot is within a reasonable range
        qpos[0] = np.clip(qpos[0]+dx, -self.size, self.size)
        qpos[1] = np.clip(qpos[1]+dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        next_obs = self._get_obs()
        reward_ctrl = np.sum(np.square(action))

        x, y = qpos[0], qpos[1]
        # reward = y * dx - x * dy
        # reward = action[0] if (y * dx - x * dy) > 0 else action[0]*0
        # reward = np.abs(action[0]) * np.sign(y * dx - x * dy)
        # delta = 0.5
        reward = (y * dx - x * dy)/np.sqrt(x**2 + y**2 + 0e-5)
        # reward = np.exp(-(np.abs(x)-self.target_radius)**2/2/delta**2)*dy - np.exp(-(np.abs(y)-self.target_radius)**2/2/delta**2)*dx
        reward /= (1 + np.abs(np.sqrt(x**2 + y**2) - self.target_radius))

        infos = {'circle_reward': reward,
                 'control_reward': reward_ctrl,
                 'action_1': action[0],
                 'action_2': action[1]}

        return next_obs, reward, False, infos

    def get_xy(self):
        qpos = self.data.qpos
        return qpos[0, 0], qpos[1, 0]

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class PointHalfCircleTest(PointHalfCircle):
    def step(self, action):
        action = np.clip(action, -self.action_clip_value, self.action_clip_value)
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        # it's better to keep the orientation within -pi to pi for reducing the scope of state space.
        if qpos[2] > np.pi:
            qpos[2] = qpos[2] - 2 * np.pi
        elif qpos[2] < - np.pi:
            qpos[2] = qpos[2] + 2 * np.pi

        # computing the increment in each direction
        ori = qpos[2]
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # ensuring that the robot is within a reasonable range
        qpos[0] = np.clip(qpos[0]+dx, -self.size, self.size)
        qpos[1] = np.clip(qpos[1]+dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        next_obs = self._get_obs()
        reward_ctrl = np.sum(np.square(action))
        done = False

        x, y = qpos[0], qpos[1]
        reward = (y * dx - x * dy)/np.sqrt(x**2 + y**2 + 0e-5)
        reward /= (1 + np.abs(np.sqrt(x**2 + y**2) - self.target_radius))
        if x < X_LOW:
            reward = 0
            done = True
            # print('terminating in true environment')

        infos = {'circle_reward': reward,
                 'control_reward': reward_ctrl,
                 'action_1': action[0],
                 'action_2': action[1]}

        return next_obs, reward, done, infos


class PointObstacle(mujoco_env.MujocoEnv):
    def __init__(
            self,
            circle_reward=True,
            start_on_circle=True,
            xml_file=ABS_PATH+'/xmls/point_circle.xml',
            size=20,
            destination=[0, 10],
            reward_ctrl_weight=0.0,
            action_clip_value=0.25,
            obstacle_position=[-2, 5, -2, 2],  #obstacle is a rectangle at  x1, x2, y1, y2
            *args,
            **kwargs
        ):

        self.size = size
        self.start_on_circle = start_on_circle
        self.destination = destination
        self.obstacle_position = obstacle_position
        self.circle_reward = circle_reward
        self.reward_ctrl_weight = reward_ctrl_weight
        self.action_clip_value = action_clip_value
        super(PointObstacle, self).__init__(xml_file, 1)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flatten(),
            self.data.qvel.flat,
            self.get_body_com('torso').flat,
        ])

    def reset_model(self):  # used for constraint learning, policy learning, and testing
        qvel = self.init_qvel
        # self.init_qpos is always 0
        starting_piont = [0, -8]
        x_scale = 1
        y_scale = 1
        qpos = self.init_qpos.copy()
        qpos[0] = qpos[0] + self.np_random.uniform(low=-x_scale, high=x_scale, size=1) + starting_piont[0]
        qpos[1] = qpos[1] + self.np_random.uniform(low=-y_scale, high=y_scale, size=1) + starting_piont[1]
        qpos[2] = self.np_random.uniform(low=-np.pi/20, high=np.pi/20) + np.pi/2

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def new_step(self, action):
        self.do_simulation(action, self.frame_skip)
        qpos = np.copy()

    def step(self, action):
        action = np.clip(action, -self.action_clip_value, self.action_clip_value)
        action[0] = np.clip(action[0], -0.0, self.action_clip_value)   # it's helpful to avoid the mini circles.
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        # it's better to keep the orientation within -pi to pi for reducing the scope of state space.
        if qpos[2] > np.pi:
            qpos[2] = qpos[2] - 2 * np.pi
        elif qpos[2] < - np.pi:
            qpos[2] = qpos[2] + 2 * np.pi

        # computing the increment in each direction
        ori = qpos[2]
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # ensuring that the robot is within a reasonable range
        qpos[0] = np.clip(qpos[0]+dx, self.obstacle_position[0], self.size) # there is a known wall to the left of the obstacle
        qpos[1] = np.clip(qpos[1]+dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        next_obs = self._get_obs()
        reward_ctrl = np.sum(np.square(action))

        x, y = qpos[0], qpos[1]

        dis2tar = np.sqrt((x-self.destination[0])**2 + (y-self.destination[1])**2)
        reward = (- dis2tar) / self.size  # distance / size of map
        if dis2tar <= 0.3:
            reward = reward + 0.1
        done = False
        # if x < self.obstacle_position[0]:  # there is a known wall to the left of the obstacle
        #     reward = -3
        #     done = False
        #     # print('terminating in true environment')
        infos = {'circle_reward': reward,
                 'control_reward': reward_ctrl,
                 'action_1': action[0],
                 'action_2': action[1]}

        return next_obs, reward, done, infos

    def get_xy(self):
        qpos = self.data.qpos
        return qpos[0, 0], qpos[1, 0]

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class PointObstacleTest(PointObstacle):
    def step(self, action):
        action = np.clip(action, -self.action_clip_value, self.action_clip_value)
        action[0] = np.clip(action[0], -0.0, self.action_clip_value)   # it's helpful to avoid the mini circles.
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        # it's better to keep the orientation within -pi to pi for reducing the scope of state space.
        if qpos[2] > np.pi:
            qpos[2] = qpos[2] - 2 * np.pi
        elif qpos[2] < - np.pi:
            qpos[2] = qpos[2] + 2 * np.pi

        # computing the increment in each direction
        ori = qpos[2]
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # ensuring that the robot is within a reasonable range
        qpos[0] = np.clip(qpos[0]+dx, self.obstacle_position[0], self.size) # there is a known wall to the left of the obstacle
        qpos[1] = np.clip(qpos[1]+dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        next_obs = self._get_obs()
        reward_ctrl = np.sum(np.square(action))
        done = False

        x, y = qpos[0], qpos[1]
        dis2tar = np.sqrt((x-self.destination[0])**2 + (y-self.destination[1])**2)
        reward = ( - dis2tar) / self.size  # distance / size of map
        if dis2tar <= 0.3:
            reward = reward + 0.1


        # if x < obstacle[0]:  # there is a known wall to the left of the obstacle
        #     reward = -3
        #     # print('terminating in true environment')
        # if obstacle[0]<=x<=obstacle[1] & obstacle[2]<=y<=obstacle[3]:
        #     reward = -0
        #     done = True

        infos = {'circle_reward': reward,
                 'control_reward': reward_ctrl,
                 'action_1': action[0],
                 'action_2': action[1]}

        return next_obs, reward, done, infos



class PointObstacle2(mujoco_env.MujocoEnv):
    def __init__(
            self,
            circle_reward=True,
            start_on_circle=True,
            xml_file=ABS_PATH+'/xmls/point_circle.xml',
            size=20,
            destination=[0, 10],
            reward_ctrl_weight=0.0,
            action_clip_value=0.25,
            obstacle_position=[-2, 5, -2, 2],  #obstacle is a rectangle at  x1, x2, y1, y2
            *args,
            **kwargs
        ):

        self.size = size
        self.start_on_circle = start_on_circle
        self.destination = destination
        self.obstacle_position = obstacle_position
        self.circle_reward = circle_reward
        self.reward_ctrl_weight = reward_ctrl_weight
        self.action_clip_value = action_clip_value
        self.starting_point_candidates = None
        super(PointObstacle2, self).__init__(xml_file, 1)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flatten(),
            self.data.qvel.flat,
            self.get_body_com('torso').flat,
        ])

    def complex_obstacle(self, pos_back, pos_front, pos_down, pos_up, cir_x, cir_y, r, obs, acs):
        return (((pos_front >= obs[0]) & (obs[0] >= pos_back) & (pos_up >= obs[1]) & (
                    obs[1] >= pos_down)) | ((obs[0] - cir_x) ** 2 + (obs[1] - cir_y) ** 2 <= r ** 2)).astype(
            np.float32)

    def reset_model(self):  # used for constraint learning, policy learning, and testing
        qvel = self.init_qvel
        # self.init_qpos is always 0
        qpos = self.init_qpos.copy()
        if self.starting_point_candidates is not None:
            qpos[0] = self.starting_point_candidates[0, 0]
            qpos[1] = self.starting_point_candidates[0, 1]
            qpos[2] = self.starting_point_candidates[0, 2]
            self.starting_point_candidates = None
        else:
            while True:
                qpos[0] = qpos[0] + self.np_random.uniform(low=-1.5, high=5.3, size=1)
                qpos[1] = qpos[1] + self.np_random.uniform(low=-5.5, high=4, size=1)
                qpos[2] = self.np_random.uniform(low=-np.pi/20, high=np.pi/20) + np.pi/2
                if self.complex_obstacle(-2, 3, -2, 2, 3, 2, 2, qpos, None) == 0:
                    break

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def new_step(self, action):
        self.do_simulation(action, self.frame_skip)
        qpos = np.copy()

    def step(self, action):
        action = np.clip(action, -self.action_clip_value, self.action_clip_value)
        action[0] = np.clip(action[0], -0.0, self.action_clip_value)   # it's helpful to avoid the mini circles.
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        # it's better to keep the orientation within -pi to pi for reducing the scope of state space.
        if qpos[2] > np.pi:
            qpos[2] = qpos[2] - 2 * np.pi
        elif qpos[2] < - np.pi:
            qpos[2] = qpos[2] + 2 * np.pi

        # computing the increment in each direction
        ori = qpos[2]
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # ensuring that the robot is within a reasonable range
        qpos[0] = np.clip(qpos[0]+dx, self.obstacle_position[0], self.size) # there is a known wall to the left of the obstacle
        qpos[1] = np.clip(qpos[1]+dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        next_obs = self._get_obs()
        reward_ctrl = np.sum(np.square(action))

        x, y = qpos[0], qpos[1]

        dis2tar = np.sqrt((x-self.destination[0])**2 + (y-self.destination[1])**2)
        reward = (- dis2tar) / self.size  # distance / size of map
        if dis2tar <= 0.3:
            reward = reward + 0.1
        done = False
        # if x < self.obstacle_position[0]:  # there is a known wall to the left of the obstacle
        #     reward = -3
        #     done = False
        #     # print('terminating in true environment')
        infos = {'circle_reward': reward,
                 'control_reward': reward_ctrl,
                 'action_1': action[0],
                 'action_2': action[1]}

        return next_obs, reward, done, infos

    def set_starting_point(self, starting_point):
        if starting_point is not None:
            if starting_point.ndim == 1:
                starting_point = starting_point.reshape(1, -1)
        self.starting_point_candidates = starting_point

    def get_xy(self):
        qpos = self.data.qpos
        return qpos[0, 0], qpos[1, 0]

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class PointObstacle2Test(PointObstacle2):
    def step(self, action):
        action = np.clip(action, -self.action_clip_value, self.action_clip_value)
        action[0] = np.clip(action[0], -0.0, self.action_clip_value)   # it's helpful to avoid the mini circles.
        qpos = np.copy(self.data.qpos)
        qpos[2] += action[1]
        # it's better to keep the orientation within -pi to pi for reducing the scope of state space.
        if qpos[2] > np.pi:
            qpos[2] = qpos[2] - 2 * np.pi
        elif qpos[2] < - np.pi:
            qpos[2] = qpos[2] + 2 * np.pi

        # computing the increment in each direction
        ori = qpos[2]
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]

        # ensuring that the robot is within a reasonable range
        qpos[0] = np.clip(qpos[0]+dx, self.obstacle_position[0], self.size) # there is a known wall to the left of the obstacle
        qpos[1] = np.clip(qpos[1]+dy, -self.size, self.size)

        self.set_state(qpos, np.copy(self.data.qvel))
        next_obs = self._get_obs()
        reward_ctrl = np.sum(np.square(action))
        done = False

        x, y = qpos[0], qpos[1]
        dis2tar = np.sqrt((x-self.destination[0])**2 + (y-self.destination[1])**2)
        reward = ( - dis2tar) / self.size  # distance / size of map
        if dis2tar <= 0.3:
            reward = reward + 0.1


        # if x < obstacle[0]:  # there is a known wall to the left of the obstacle
        #     reward = -3
        #     # print('terminating in true environment')
        # if obstacle[0]<=x<=obstacle[1] & obstacle[2]<=y<=obstacle[3]:
        #     reward = -0
        #     done = True

        infos = {'circle_reward': reward,
                 'control_reward': reward_ctrl,
                 'action_1': action[0],
                 'action_2': action[1]}

        return next_obs, reward, done, infos


# PointDS is nearly the same as PointEllip, with slight difference in the obstacle
# configuration and starting points.
class _PointPlanarBase(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, destination, action_clip_value, done_threshold):
        self.destination = np.array(destination, dtype=np.float32)
        self.action_clip_value = action_clip_value
        self.done_threshold = done_threshold
        self.starting_point = None
        self.starting_point_candidates = None
        self.starting_point_idx = 0

        # Keep the same effective dynamics step as point_ds.xml.
        self.dt = 0.02
        self._init_qpos = np.zeros(3, dtype=np.float32)
        self._init_qvel = np.zeros(3, dtype=np.float32)
        self._qpos = self._init_qpos.copy()
        self._qvel = self._init_qvel.copy()
        self.last_obs = self._qpos.copy()

        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-2.0, -2.0], dtype=np.float32),
            high=np.array([2.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.seed()

    @property
    def data(self):
        class _Data:
            def __init__(self, qpos, qvel):
                self.qpos = qpos
                self.qvel = qvel
        return _Data(self._qpos, self._qvel)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_state(self, qpos, qvel):
        self._qpos = np.array(qpos, dtype=np.float32).copy()
        self._qvel = np.array(qvel, dtype=np.float32).copy()

    def _get_obs(self):
        return self._qpos[:2].copy()

    def _sample_starting_point(self):
        raise NotImplementedError

    def reset_model(self):
        qvel = self._init_qvel.copy()
        qpos = self._init_qpos.copy()
        starting_point, noise_scale = self._sample_starting_point()
        qpos[0] = self.np_random.uniform(low=-noise_scale, high=noise_scale) + starting_point[0]
        qpos[1] = self.np_random.uniform(low=-noise_scale, high=noise_scale) + starting_point[1]
        qpos[2] = 0.0
        self.last_obs = qpos.copy()
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self, **kwargs):
        return self.reset_model()

    def _apply_action(self, action):
        qpos = self._qpos.copy()
        qpos[2] = np.arctan2(action[1], action[0])
        qpos[0] = qpos[0] + action[0] * self.dt
        qpos[1] = qpos[1] + action[1] * self.dt
        self.set_state(qpos, self._qvel.copy())
        self.last_obs = qpos.copy()
        return qpos

    def _postprocess_action(self, action):
        return action

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.action_clip_value, self.action_clip_value)
        action = self._postprocess_action(action)
        qpos = self._apply_action(action)
        next_obs = self._get_obs()
        reward_ctrl = float(np.sum(np.square(action)))

        x, y = qpos[0], qpos[1]
        dis2tar = float(np.sqrt((x - self.destination[0]) ** 2 + (y - self.destination[1]) ** 2))
        reward = -dis2tar
        done = dis2tar < self.done_threshold
        if done:
            reward += self._success_bonus()

        infos = {
            "circle_reward": reward,
            "control_reward": reward_ctrl,
            "action_1": float(action[0]),
            "action_2": float(action[1]),
        }
        return next_obs, reward, done, infos

    def _success_bonus(self):
        return 0.0

    def set_starting_point(self, starting_point):
        if starting_point is not None and starting_point.ndim == 1:
            starting_point = starting_point.reshape(1, -1)
        self.starting_point_candidates = starting_point

    def viewer_setup(self):
        return

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.zeros((240, 320, 3), dtype=np.uint8)
        return None


class PointDS(_PointPlanarBase):
    def __init__(
            self,
            xml_file=ABS_PATH+'/xmls/point_ds.xml',
            destination=[0, 0],
            action_clip_value=np.inf,
            obstacle_geom=[-2, 5, -2, 2],  # obstacle is a rectangle at x1, x2, y1, y2
            *args,
            **kwargs
        ):
        del xml_file, args, kwargs
        self.obstacle_geom = obstacle_geom
        super(PointDS, self).__init__(destination, action_clip_value, done_threshold=0.02)

    def _sample_starting_point(self):
        if self.starting_point_candidates is not None:
            starting_point = self.starting_point_candidates[
                self.starting_point_idx % self.starting_point_candidates.shape[0]
            ]
            noise_scale = 0.0
            self.starting_point_idx += 1
        else:
            starting_point = np.array([1.2, 1.2], dtype=np.float32)
            noise_scale = 0.7
        return starting_point, noise_scale

class PointDSTest(PointDS):
    def __init__(
            self,
            xml_file=ABS_PATH+'/xmls/point_ds.xml',
            destination=[0, 0],
            action_clip_value=np.inf,
            obstacle_geom=[-2, 5, -2, 2],  #obstacle is a rectangle at  x1, x2, y1, y2
            *args,
            **kwargs
        ):
        mean = np.array([[1, 1]])
        cov = np.array([[[2, -1.5], [-1.5, 3]]])/70
        self.set_true_gmm_constraint(mean, cov)
        super(PointDSTest, self).__init__(
            xml_file=xml_file,
            destination=destination,
            action_clip_value=action_clip_value,
            obstacle_geom=obstacle_geom,
            *args,
            **kwargs
        )

    def step(self, action):
        next_obs, reward, done, infos = super().step(action)
        infos['cost'] = self.true_cost_function(next_obs, action)
        return next_obs, reward, done, infos

    def true_cost_function(self, obs, acs):
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        # Define true obstacle using GMM
        obs = obs - self.gmm_true.means_[0]
        cov_inv = np.linalg.inv(self.gmm_true.covariances_[0])
        cost = np.zeros([obs.shape[0]])
        for i in range(obs.shape[0]):
            value = (obs[i] @ cov_inv @ obs[i].T).item()
            cost[i] = (value <= 5)

        if obs.shape[0] == 1:
            return cost[0]
        else:
            return cost

    def set_true_gmm_constraint(self, mean, cov, confidence=5):
        from types import SimpleNamespace
        gmm = SimpleNamespace()
        gmm.means_ = mean
        gmm.covariances_ = cov
        gmm.n_components = 1
        gmm.confidence = confidence
        self.gmm_true = gmm

    @property
    def get_gmm(self):
        return self.gmm_true

class PointEllip(_PointPlanarBase):
    def __init__(
            self,
            xml_file=ABS_PATH+'/xmls/point_ds.xml',
            destination=[0, 0],
            action_clip_value=np.inf,
            *args,
            **kwargs
        ):
        del xml_file, args, kwargs
        super(PointEllip, self).__init__(destination, action_clip_value, done_threshold=0.03)

    def _sample_starting_point(self):
        if self.starting_point_candidates is not None:
            starting_point = self.starting_point_candidates[
                self.starting_point_idx % self.starting_point_candidates.shape[0]
            ]
            noise_scale = 0.0
            self.starting_point_idx += 1
        else:
            starting_point = np.array([0.9, 1.0], dtype=np.float32)
            noise_scale = 0.5
        return starting_point, noise_scale

    def _postprocess_action(self, action):
        speeds = np.linalg.norm(action)
        if speeds > 2:
            action = action / speeds * 2.0
        return action

    def _success_bonus(self):
        return 100.0

class PointEllipTest(PointEllip):
    def __init__(
            self,
            xml_file=ABS_PATH+'/xmls/point_ds.xml',
            destination=[0, 0],
            action_clip_value=np.inf,
            *args,
            **kwargs
        ):
        mean = np.array([[0.82, 1], [0.72, 0.55]])
        cov = np.array([[[1.3, -0.8], [-0.8, 1.8]],
                        [[1., 0], [0, 1.]]])/80
        self.set_true_gmm_constraint(mean, cov)
        super(PointEllipTest, self).__init__(
            xml_file=xml_file,
            destination=destination,
            action_clip_value=action_clip_value,
            *args,
            **kwargs
        )

    def step(self, action):
        next_obs, reward, done, infos = super().step(action)
        infos['cost'] = self.true_cost_function(next_obs, action)
        return next_obs, reward, done, infos

    def true_cost_function(self, obs, acs):
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        # Define true obstacle using GMM
        cost = np.zeros([obs.shape[0]])

        for j in range(self.gmm_true.n_components):
            obs_ = obs - self.gmm_true.means_[j]
            cov_inv = np.linalg.inv(self.gmm_true.covariances_[j])
            for i in range(obs_.shape[0]):
                value = (obs_[i] @ cov_inv @ obs_[i].T).item()
                cost[i] = max(cost[i], (value <= self.gmm_true.confidence))

        if obs.shape[0] == 1:
            return cost[0]
        else:
            return cost

    def set_true_gmm_constraint(self, mean, cov, confidence=5):
        from types import SimpleNamespace
        gmm = SimpleNamespace()
        gmm.means_ = mean
        gmm.covariances_ = cov
        gmm.n_components = mean.shape[0]
        gmm.confidence = confidence
        self.gmm_true = gmm

    @property
    def get_gmm(self):
        return self.gmm_true

if __name__=='__main__':
    a = PointDSTest()
    print(a.reset(starting_point=[1,1], noise_scale=1))
