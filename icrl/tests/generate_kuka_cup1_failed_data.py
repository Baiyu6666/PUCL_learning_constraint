"""Save one pass of higher-return, unmodulated KukaCup1 nominal rollouts."""

import argparse
import json
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gym
import gym_panda  # Register KukaCup1-v0.
import numpy as np

from icrl.ds_policy import DS_Policy


def generate(dataset_dir=None):
    root = Path(dataset_dir) if dataset_dir else Path(__file__).resolve().parents[1] / 'expert_data/KukaCup1'
    expert_dir = root / 'files/EXPERT/rollouts'
    expert_files = sorted(expert_dir.glob('*.pkl'), key=lambda path: int(path.stem))
    if not expert_files:
        raise ValueError(f'No expert rollouts found in {expert_dir}')
    output = root / 'files/FAILED/rollouts'
    if any(output.glob('*.pkl')):
        raise FileExistsError(f'Failed rollouts already exist in {output}')

    policy = DS_Policy(3, lambda obs: 2.0 * obs[3:6], stable_modulation_basis=True)
    env = gym.make('KukaCup1-v0', config_path=str(root / 'task_config.json'))
    records, saved = [], []
    try:
        # Exactly one rollout per expert task; no constraint training or modulation.
        for expert_file in expert_files:
            with expert_file.open('rb') as stream:
                expert = pickle.load(stream)
            expert_return = float(np.asarray(expert['rewards']).sum())
            env.set_starting_point(expert['observations'][0])
            obs = env.reset()
            observations, actions, step_rewards = [], [], []
            done = False
            while not done:
                action, _ = policy.predict(obs[None, :], deterministic=True)
                observations.append(obs.copy())
                actions.append(action[0].copy())
                obs, reward, done, info = env.step(action[0])
                step_rewards.append(float(reward))

            nominal_return = float(np.sum(step_rewards))
            keep = nominal_return > expert_return
            record = {
                'source_expert_file': expert_file.name,
                'source_demo': expert['source_demo'],
                'expert_return': expert_return,
                'nominal_return': nominal_return,
                'return_improvement': nominal_return - expert_return,
                'steps': len(actions),
                'goal_reached': bool(info['goal_reached']),
                'path_length': float(info['path_length']),
                'saved': keep,
                'rollout_file': expert_file.name if keep else None,
            }
            records.append(record)
            if keep:
                saved.append((expert_file.name, {
                    'observations': np.asarray(observations, dtype=np.float32),
                    'actions': np.asarray(actions, dtype=np.float32),
                    'rewards': np.array([nominal_return], dtype=np.float64),
                    'lengths': np.array([len(actions)], dtype=np.int64),
                    'save_scheme': 'FAILED',
                    'source_demo': expert['source_demo'],
                    'source_expert_file': expert_file.name,
                    'expert_return': expert_return,
                    'goal': expert['observations'][0, :3] + expert['observations'][0, 3:6],
                    'terminal_observation': obs.copy(),
                    'step_rewards': np.asarray(step_rewards, dtype=np.float64),
                    'generation': 'initial_unmodulated_nominal_ds',
                }))
    finally:
        env.close()

    output.mkdir(parents=True, exist_ok=True)
    for filename, rollout in saved:
        with (output / filename).open('wb') as stream:
            pickle.dump(rollout, stream, protocol=4)
    summary = {
        'environment': 'KukaCup1-v0',
        'sampling_passes': 1,
        'constraint_updates': 0,
        'nominal_ds': '2 * (goal - position), with DS_Policy velocity normalization',
        'selection': 'nominal episode return > corresponding expert episode return',
        'failed_label': 'Trajectory-level higher-return nominal demonstration; not a per-state infeasibility label.',
        'candidate_trajectories': len(records),
        'saved_trajectories': len(saved),
        'total_transitions': sum(int(rollout['lengths'][0]) for _, rollout in saved),
        'trajectories': records,
    }
    (output.parent / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(generate(args.dataset_dir), indent=2))
