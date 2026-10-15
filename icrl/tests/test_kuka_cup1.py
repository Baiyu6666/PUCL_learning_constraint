import json
import pickle
import tempfile
import unittest
from pathlib import Path

import gym
import numpy as np
import gym_panda
from gym_panda.envs.KukaCup1 import KukaCup1, DEFAULT_CONFIG
from icrl.tests.prepare_kuka_cup1_data import resample_by_arc_length


class KukaCup1Tests(unittest.TestCase):
    def make_env(self, **overrides):
        options = dict(obstacle_center=[0.5, 0.5], obstacle_radius=0.1,
                       workspace_low=[0, 0, 0], workspace_high=[1, 1, 1],
                       start=[0.2, 0.2, 0.2], goal=[0.8, 0.2, 0.2],
                       step_size=0.1, goal_tolerance=0.005, max_steps=20)
        options.update(overrides)
        return gym.make('KukaCup1-v0', **options)

    def test_registration_and_position_dynamics(self):
        env = self.make_env()
        try:
            obs = env.reset()
            self.assertTrue(env.observation_space.contains(obs))
            obs, reward, done, info = env.step([1, 0, 0])
            np.testing.assert_allclose(obs[:3], [0.3, 0.2, 0.2], atol=1e-7)
            np.testing.assert_allclose(obs[3:6], [0.5, 0, 0], atol=1e-7)
            self.assertAlmostEqual(reward, -0.1)
            self.assertFalse(done)
        finally:
            env.close()

    def test_cylinder_ignores_height_and_action(self):
        env = self.make_env()
        try:
            obs = np.array([[0.5, 0.5, -100], [0.5, 0.5, 100], [0.7, 0.5, 0.5]])
            np.testing.assert_array_equal(env.true_cost_function(obs, np.zeros((3, 3))), [1, 1, 0])
            np.testing.assert_array_equal(env.true_cost_function(obs, np.ones((3, 3))), [1, 1, 0])
        finally:
            env.close()

    def test_obstacle_is_not_a_physical_wall(self):
        env = self.make_env(start=[0.35, 0.5, 0.2], goal=[0.8, 0.5, 0.2])
        try:
            env.reset()
            obs, _, _, info = env.step([1, 0, 0])
            self.assertAlmostEqual(float(obs[0]), 0.45, places=6)
            self.assertEqual(info['cost'], 1)
        finally:
            env.close()

    def test_success_return_is_negative_path_length(self):
        env = self.make_env()
        try:
            env.reset()
            total = 0
            for _ in range(6):
                _, reward, done, info = env.step([1, 0, 0])
                total += reward
            self.assertTrue(done)
            self.assertTrue(info['is_success'])
            self.assertAlmostEqual(total, -0.6)
            self.assertAlmostEqual(total, -info['path_length'])
        finally:
            env.close()

    def test_idle_timeout_is_worse_than_reaching(self):
        env = self.make_env(max_steps=2)
        try:
            env.reset()
            _, first, done, _ = env.step([0, 0, 0])
            self.assertFalse(done)
            _, last, done, info = env.step([0, 0, 0])
            self.assertTrue(done)
            self.assertFalse(info['goal_reached'])
            self.assertTrue(info['TimeLimit.truncated'])
            self.assertLess(first + last, -2 * 0.1 * np.sqrt(3))
        finally:
            env.close()

    def test_demo_start_preserves_individual_goal(self):
        env = self.make_env()
        try:
            starts = np.array([[0.2, 0.2, 0.2, 0.6, 0, 0, 0, 0, 0],
                               [0.8, 0.2, 0.2, -0.6, 0, 0, 0, 0, 0]])
            env.set_starting_point(starts)
            first, second = env.reset(), env.reset()
            np.testing.assert_allclose(first[:6], starts[0, :6], atol=1e-7)
            np.testing.assert_allclose(second[:6], starts[1, :6], atol=1e-7)
        finally:
            env.close()

    def test_missing_geometry_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'task_config.json'
            path.write_text(json.dumps({'obstacle_center': None, 'obstacle_radius': None}))
            with self.assertRaisesRegex(ValueError, 'obstacle_center'):
                KukaCup1(config_path=path)

    def test_default_resets_cycle_over_demo_tasks(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        env = gym.make('KukaCup1-v0', obstacle_center=[0.6, -0.08], obstacle_radius=0.04)
        try:
            for task in config['tasks']:
                obs = env.reset()
                np.testing.assert_allclose(obs[:3], task['start'], atol=1e-7)
                np.testing.assert_allclose(obs[:3] + obs[3:6], task['goal'], atol=1e-7)
        finally:
            env.close()

    def test_converted_demonstrations_match_dynamics_and_reward(self):
        root = DEFAULT_CONFIG.parent
        config = json.loads(DEFAULT_CONFIG.read_text())
        self.assertEqual(len(config['tasks']), 10)
        for task in config['tasks']:
            with (root / 'files/EXPERT/rollouts' / task['rollout_file']).open('rb') as stream:
                rollout = pickle.load(stream)
            observations, actions = rollout['observations'], rollout['actions']
            self.assertEqual(len(observations), int(rollout['lengths'][0]))
            self.assertEqual(actions.shape, (len(observations), 3))
            np.testing.assert_allclose(observations[:, :3] + observations[:, 3:6],
                                       np.broadcast_to(task['goal'], (len(observations), 3)), atol=1e-7)
            next_positions = np.vstack([observations[1:, :3], rollout['terminal_observation'][:3]])
            displacements = next_positions - observations[:, :3]
            np.testing.assert_allclose(displacements, actions * config['step_size'], atol=1e-7)
            self.assertAlmostEqual(float(rollout['rewards'][0]), -np.linalg.norm(displacements, axis=1).sum(), places=5)

    def test_arc_length_sampling_removes_time_density_bias(self):
        positions = np.array([[0., 0., 0.], [0., 0., 0.], [0.1, 0., 0.], [1., 0., 0.]])
        sampled, times = resample_by_arc_length(positions, np.array([0., 1., 2., 3.]), 6)
        np.testing.assert_allclose(sampled[:, 0], np.linspace(0, 1, 6))
        np.testing.assert_array_equal(sampled[[0, -1]], positions[[0, -1]])
        self.assertTrue(np.all(np.diff(times) > 0))


if __name__ == '__main__':
    unittest.main()
