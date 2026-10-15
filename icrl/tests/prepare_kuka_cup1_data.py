"""Convert the selected raw iiwa7 segments into PUCL rollout dictionaries.

Run from the project root: python icrl/tests/prepare_kuka_cup1_data.py
Actions are Cartesian displacements divided by the simulated dynamics step,
not recorded robot commands. Original joint angles and poses remain in raw/.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def resample_by_arc_length(positions, times, num_points):
    """Interpolate equally spaced positions while retaining both endpoints."""
    if num_points < 2:
        raise ValueError('num_points must be at least 2')
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    unique_arc, indices = np.unique(arc, return_index=True)
    if len(unique_arc) < 2:
        raise ValueError('cannot resample a stationary trajectory')
    targets = np.linspace(0.0, unique_arc[-1], num_points)
    sampled = np.column_stack([np.interp(targets, unique_arc, positions[indices, dimension])
                               for dimension in range(3)])
    sampled_times = np.interp(targets, unique_arc, times[indices])
    sampled[0], sampled[-1] = positions[0], positions[-1]
    sampled_times[0], sampled_times[-1] = times[0], times[-1]
    return sampled, sampled_times


def prepare(dataset_dir=None, num_points=100):
    root = Path(dataset_dir) if dataset_dir else Path(__file__).resolve().parents[1] / "expert_data/KukaCup1"
    manifest = json.loads((root / "manifest.json").read_text())
    config_file = root / "task_config.json"
    config = json.loads(config_file.read_text()) if config_file.exists() else {
        "coordinate_frame": "iiwa7_link_0",
        "units": "metres",
        "obstacle_center": [0.60, -0.08],
        "obstacle_radius": 0.04,
        "geometry_source": "Estimated evaluation cylinder, not measured geometry.",
        "workspace_low": [0.3, -0.4, 0.1],
        "workspace_high": [0.9, 0.3, 0.6],
        "step_size": 0.008,
        "goal_tolerance": 0.01,
        "max_steps": 200,
    }
    step_size = config["step_size"]
    rollouts = root / "files/EXPERT/rollouts"
    rollouts.mkdir(parents=True, exist_ok=True)
    tasks = []
    for index, demo in enumerate(manifest["selected_demo_ids"]):
        with np.load(root / "raw" / (demo + ".npz"), allow_pickle=False) as data:
            positions = data["poses"][:, :3].copy()
            times = data["t"].copy()
        goal = positions[-1].copy()
        # End when the simulation would first satisfy its goal tolerance.
        hits = np.flatnonzero(np.linalg.norm(positions - goal, axis=1) <= config["goal_tolerance"])
        stop = int(hits[0])
        if stop < 1:
            raise ValueError(f"{demo}: start is already at the goal")
        positions, times = positions[:stop + 1], times[:stop + 1]
        source_num_positions = len(positions)
        source_path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        positions, times = resample_by_arc_length(positions, times, num_points)
        displacements = np.diff(positions, axis=0)
        actions = displacements / step_size
        if np.any(np.abs(actions) > 1.0 + 1e-6):
            raise ValueError(f"{demo}: source transitions exceed action bounds; resample explicitly")
        if np.any(positions < config["workspace_low"]) or np.any(positions > config["workspace_high"]):
            raise ValueError(f"{demo}: source positions exceed the configured workspace")
        previous_displacements = np.vstack([np.zeros(3), displacements[:-1]])
        observations = np.concatenate([positions[:-1], goal - positions[:-1], previous_displacements], axis=1)
        path_length = float(np.linalg.norm(displacements, axis=1).sum())
        task = {"demo_id": demo, "rollout_file": f"{index}.pkl", "start": positions[0].tolist(),
                "goal": goal.tolist(), "terminal_position": positions[-1].tolist(),
                "num_transitions": len(actions), "num_positions": len(positions),
                "source_num_positions": source_num_positions,
                "source_path_length": source_path_length,
                "path_length": path_length}
        rollout = {
            "observations": observations.astype(np.float32),
            "actions": actions.astype(np.float32),
            "rewards": np.array([-path_length], dtype=np.float64),
            "lengths": np.array([len(actions)], dtype=np.int64),
            "save_scheme": "EXPERT",
            "source_demo": demo,
            "goal": goal,
            "terminal_observation": np.concatenate([positions[-1], goal - positions[-1], displacements[-1]]).astype(np.float32),
            "timestamps": times,
            "action_definition": "delta_position / step_size; derived, not measured robot command",
            "downsampling": {"method": "uniform_arc_length", "num_positions": num_points},
        }
        with (rollouts / task["rollout_file"]).open("wb") as stream:
            pickle.dump(rollout, stream, protocol=4)
        tasks.append(task)
    config["tasks"] = tasks
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    manifest["conversion_status"] = "pucl_rollouts_ready"
    manifest["converted_rollouts"] = tasks
    manifest["downsampling"] = {"method": "uniform_arc_length", "num_positions_per_demo": num_points}
    manifest["total_training_observations"] = sum(task['num_transitions'] for task in tasks)
    manifest["notes"] = [
        "raw/ arrays are unchanged; converted rollouts use Cartesian position only.",
        "Goals are individual demonstration endpoints. Each converted trajectory stops at its first goal-tolerance hit.",
        f"Converted trajectories are resampled to {num_points} positions by uniform arc length, including both endpoints.",
        "Actions are derived from displacement / step_size; no measured action or reward exists in the source.",
        "Rewards are negative accumulated Cartesian path length in metres.",
        "The default evaluation cylinder is estimated; its geometry is not used as training labels.",
    ]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', type=int, default=100, help='number of positions including the terminal point')
    args = parser.parse_args()
    tasks = prepare(num_points=args.points)
    print(f"Prepared {len(tasks)} KukaCup1 expert rollouts")
    for task in tasks:
        print(task["demo_id"], task["rollout_file"], task["num_transitions"], task["path_length"])
