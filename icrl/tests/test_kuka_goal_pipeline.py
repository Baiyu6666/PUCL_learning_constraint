import unittest
from pathlib import Path

import gym
import gym_panda
import numpy as np

from icrl.constraint_net import ConstraintNet
from icrl.ds_policy import DS_Policy
from icrl.goal_reaching_evaluation import evaluate_goal_reaching_policy
from icrl.utils import load_expert_data, sample_from_same_starting_points_as_demonstrations
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalizeWithCost


class KukaGoalPipelineTests(unittest.TestCase):
    def make_vector_env(self, **overrides):
        options = dict(obstacle_center=[0.5, 0.5], obstacle_radius=0.1,
                       workspace_low=[0, 0, 0], workspace_high=[1, 1, 1],
                       start=[0.2, 0.2, 0.2], goal=[0.8, 0.2, 0.2],
                       step_size=0.1, goal_tolerance=0.01, max_steps=100)
        options.update(overrides)
        return VecNormalizeWithCost(DummyVecEnv([lambda: gym.make('KukaCup1-v0', **options)]),
                                    norm_obs=False, norm_reward=False, norm_cost=False)

    def test_sampling_retains_forward_and_return_goals(self):
        env = self.make_vector_env()
        try:
            starts = np.array([[0.2, 0.2, 0.2, 0.6, 0, 0, 0, 0, 0],
                               [0.8, 0.2, 0.2, -0.6, 0, 0, 0, 0, 0]])
            policy = DS_Policy(3, lambda x: 2 * x[3:6], stable_modulation_basis=True)
            observations, _, actions, rewards, lengths = sample_from_same_starting_points_as_demonstrations(
                policy, env, starts, np.array([1, 1]), cost_function=lambda obs, action: np.zeros(len(obs)))
            self.assertEqual(len(lengths), 2)
            index = 0
            for task_index, length in enumerate(lengths):
                trajectory = observations[index:index + length]
                goal = starts[task_index, :3] + starts[task_index, 3:6]
                np.testing.assert_allclose(trajectory[:, :3] + trajectory[:, 3:6],
                                           np.broadcast_to(goal, (length, 3)), atol=1e-6)
                self.assertAlmostEqual(rewards[task_index],
                                       -np.linalg.norm(actions[index:index + length] * 0.1, axis=1).sum(), places=5)
                index += length
        finally:
            env.close()

    def test_episode_returns_are_not_step_averages(self):
        env = self.make_vector_env()
        try:
            starts = np.array([[0.2, 0.2, 0.2, 0.1, 0, 0, 0, 0, 0],
                               [0.8, 0.2, 0.2, -0.3, 0, 0, 0, 0, 0]])
            env.env_method('set_starting_point', starts)
            policy = DS_Policy(3, lambda x: np.sign(x[3:6]))
            result = evaluate_goal_reaching_policy(policy, env, env.get_attr('true_cost_function')[0], 4)
            self.assertAlmostEqual(result['average_reward'], -0.2, places=6)
            self.assertAlmostEqual(result['average_episode_length'], 2.0)
            self.assertEqual(result['goal_reached_portion'], 1.0)
            expected_goals = [starts[i % 2, :3] + starts[i % 2, 3:6] for i in range(4)]
            np.testing.assert_allclose([r['goal'] for r in result['episodes']], expected_goals, atol=1e-6)
        finally:
            env.close()

    def test_terminal_cost_uses_finished_task_not_auto_reset(self):
        env = self.make_vector_env()
        try:
            starts = np.array([[0.4, 0.5, 0.2, 0.1, 0, 0, 0, 0, 0],
                               [0.2, 0.2, 0.2, 0.1, 0, 0, 0, 0, 0]])
            env.env_method('set_starting_point', starts)
            policy = DS_Policy(3, lambda x: np.sign(x[3:6]))
            result = evaluate_goal_reaching_policy(policy, env, env.get_attr('true_cost_function')[0], 2)
            self.assertFalse(result['episodes'][0]['safe'])
            self.assertEqual(result['episodes'][0]['violating_steps'], 1)
            self.assertTrue(result['episodes'][1]['safe'])
            self.assertEqual(result['safe_goal_reached_portion'], 0.5)
        finally:
            env.close()

    def test_position_network_does_not_depend_on_goal(self):
        model = ConstraintNet(9, 3, [8], None, None, is_discrete=False,
                              optimizer_class=None, obs_select_dim=[0, 1, 2], acs_select_dim=[-1])
        observations = np.array([[0.6, -0.1, 0.3, 0.2, 0.1, 0.2, 0, 0, 0],
                                 [0.6, -0.1, 0.3, -0.3, -0.2, -0.1, 0, 0, 0]])
        costs = model.cost_function_non_binary(observations, np.zeros((2, 3)))
        self.assertEqual(float(costs[0]), float(costs[1]))
        _, grads = model.build_gamma_with_grad_for_ds(observations, np.zeros((2, 3)))
        np.testing.assert_allclose(grads[0], grads[1])

    def test_modulation_handles_axis_aligned_cylinder_normal(self):
        policy = DS_Policy(3, lambda x: x[3:6], stable_modulation_basis=True)
        policy.modulate_with_NN(lambda obs, action: (np.array([2.0]), np.array([1.0, 0, 0])), rho=1)
        action, _ = policy.predict(np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0]]), velocity_normalization=False)
        np.testing.assert_allclose(action, [[0.5, 0, 0]])

    def test_task_selection_is_reproducible_without_reseeding_training(self):
        root = Path(__file__).resolve().parents[1] / 'expert_data/KukaCup1'
        np.random.seed(7)
        expected = np.random.RandomState(7).rand()
        first = load_expert_data(str(root), 4, selection_seed=23)
        self.assertEqual(np.random.rand(), expected)
        second = load_expert_data(str(root), 4, selection_seed=23)
        np.testing.assert_array_equal(first[0][0], second[0][0])


if __name__ == '__main__':
    unittest.main()
