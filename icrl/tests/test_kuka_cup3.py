"""Dataset replay and evaluation without known geometry for KukaCup3."""

import hashlib
import json
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import gym
import gym_panda
import numpy as np

from icrl.ds_policy import DS_Policy
from icrl.goal_reaching_evaluation import evaluate_goal_reaching_policy
from icrl.tests.prepare_kuka_cup3_data import refresh_metadata
from stable_baselines3.common.vec_env import DummyVecEnv


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'icrl/expert_data/KukaCup3'


class KukaCup3Tests(unittest.TestCase):
    def test_metadata_refresh_follows_curated_files_without_rewriting_them(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'dataset'
            shutil.copytree(DATASET, root)
            (root / 'files/EXPERT/rollouts/0.pkl').unlink()
            before = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (root / 'files').rglob('*.pkl')}
            tasks = refresh_metadata(root)
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(len(tasks), 24)
            self.assertEqual(manifest['num_demonstrations'], 24)
            self.assertEqual(manifest['total_training_observations'], 24 * 99)
            self.assertEqual(manifest['num_failed_demonstrations'], 15)
            self.assertNotIn('0.pkl', [task['rollout_file'] for task in tasks])
            for relative, digest in before.items():
                self.assertEqual(hashlib.sha256((root / relative).read_bytes()).hexdigest(), digest)

    def test_wide_workspace_does_not_clip_full_horizon_motion(self):
        env = gym.make('KukaCup1-v0', config_path=str(DATASET / 'task_config.json'))
        try:
            # Start below the former -0.45 y limit and move away from the plot box.
            start = np.array([0.55, -0.47, 0.45])
            env.set_starting_point(np.r_[start, [0.1, 0, 0], [0, 0, 0]])
            env.reset()
            action = np.array([1., -1., 1.])
            for step in range(1, env.max_steps + 1):
                obs, _, done, _ = env.step(action)
                np.testing.assert_allclose(obs[:3], start + step * env.step_size * action, atol=1e-6)
                self.assertEqual(done, step == env.max_steps)
            self.assertLess(obs[1], env.plot_workspace_low[1])
            self.assertGreater(obs[2], env.plot_workspace_high[2])
        finally:
            env.close()

    def test_imported_rollouts_replay_and_keep_raw_checksums(self):
        manifest = json.loads((DATASET / 'manifest.json').read_text())
        self.assertEqual(manifest['num_demonstrations'], 25)
        self.assertEqual(manifest['total_training_observations'], 2475)
        self.assertEqual(manifest['num_failed_demonstrations'], 15)
        env = gym.make('KukaCup1-v0', config_path=str(DATASET / 'task_config.json'))
        try:
            for demo in manifest['demonstrations']:
                self.assertEqual(hashlib.sha256((DATASET / demo['file']).read_bytes()).hexdigest(), demo['sha256'])
                with (DATASET / demo['rollout_file']).open('rb') as stream:
                    rollout = pickle.load(stream)
                self.assertEqual(rollout['source_demo'], demo['demo_id'])
                self.assertEqual(hashlib.sha256((DATASET / demo['rollout_file']).read_bytes()).hexdigest(),
                                 demo['rollout_sha256'])
                env.set_starting_point(rollout['observations'][0])
                obs = env.reset()
                episode_return = 0.0
                for index, action in enumerate(rollout['actions']):
                    np.testing.assert_allclose(obs, rollout['observations'][index], atol=1e-6)
                    obs, reward, done, info = env.step(action)
                    episode_return += reward
                    self.assertEqual(done, index == len(rollout['actions']) - 1)
                self.assertTrue(info['goal_reached'])
                self.assertIsNone(info['is_success'])
                self.assertAlmostEqual(episode_return, rollout['rewards'][0], places=6)
                np.testing.assert_allclose(obs, rollout['terminal_observation'], atol=1e-6)
        finally:
            env.close()

    def test_unknown_geometry_does_not_report_safety(self):
        env = DummyVecEnv([lambda: gym.make('KukaCup1-v0', config_path=str(DATASET / 'task_config.json'))])
        try:
            policy = DS_Policy(3, lambda obs: 2 * obs[3:6], stable_modulation_basis=True)
            result = evaluate_goal_reaching_policy(policy, env, None, 25)
            self.assertEqual(result['goal_reached_portion'], 1.0)
            for key in ('safe_portion', 'safe_goal_reached_portion', 'feasible_reward', 'violation_steps_portion'):
                self.assertIsNone(result[key])
            for record in result['episodes']:
                self.assertIsNone(record['safe'])
                self.assertIsNone(record['violating_steps'])
            json.dumps(result, allow_nan=False)
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                env.envs[0].true_cost_function(np.zeros((1, 9)), np.zeros((1, 3)))
        finally:
            env.close()

    def test_explicit_dataset_does_not_change_kuka_cup1_defaults(self):
        first = gym.make('KukaCup1-v0', config_path=str(DATASET / 'task_config.json'))
        original = gym.make('KukaCup1-v0')
        try:
            self.assertFalse(first.true_constraint_available)
            self.assertTrue(original.true_constraint_available)
            self.assertEqual(len(first.default_tasks), 25)
            self.assertEqual(len(original.default_tasks), 10)
        finally:
            first.close()
            original.close()


if __name__ == '__main__':
    unittest.main()
