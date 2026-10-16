"""Exercise non-MuJoCo paths in fresh interpreters with MuJoCo blocked."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
BLOCKER = '''
import importlib.abc
import sys

attempts = []

class NoMuJoCo(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in {'mujoco', 'mujoco_py', 'dm_control'}
                or fullname == 'gym.envs.mujoco' or fullname.startswith('gym.envs.mujoco.')):
            attempts.append(fullname)
            raise ModuleNotFoundError('MuJoCo deliberately unavailable: ' + fullname, name=fullname)

sys.meta_path.insert(0, NoMuJoCo())
'''


class OptionalMuJoCoTests(unittest.TestCase):
    def run_isolated(self, code):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'sitecustomize.py').write_text(textwrap.dedent(BLOCKER))
            env = dict(os.environ, MPLBACKEND='Agg', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            env['PYTHONPATH'] = os.pathsep.join(
                [directory, str(ROOT), str(ROOT / 'custom_envs'), env.get('PYTHONPATH', '')])
            result = subprocess.run(
                [sys.executable, '-c', textwrap.dedent(code)], cwd=str(ROOT),
                env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_training_entries_and_shared_utilities_do_not_import_mujoco(self):
        self.run_isolated('''
            import sitecustomize
            import numpy as np
            from types import SimpleNamespace
            import custom_envs.envs as envs
            import custom_envs.envs.utils
            import icrl.dscl
            import icrl.pucl
            from icrl.true_constraint_net import bridges

            # Importing utilities must not load any simulator environment classes.
            assert 'AntTest' in dir(envs)
            assert 'custom_envs.envs.ant' not in __import__('sys').modules
            try:
                envs.missing_environment
            except AttributeError:
                pass
            else:
                raise AssertionError('Unknown environment attribute was accepted')

            env = SimpleNamespace(normalize=False,
                action_map_dict={0: np.array([1., 0.])},
                constraint_regions=[(np.array([.25, -.25]), .5, .5)])
            np.testing.assert_array_equal(bridges(env, np.array([[0., 0.]]), np.array([0])), [1])
            assert not sitecustomize.attempts, sitecustomize.attempts
        ''')

    def test_pybullet_reach_and_available_kuka_task_work_without_mujoco(self):
        self.run_isolated('''
            import sitecustomize
            import gym
            import gym_panda
            from icrl.ds_policy import DS_Policy

            # Older branches have Reach tasks but do not include KukaCup1 yet.
            if any(spec.id == 'KukaCup1-v0' for spec in gym.envs.registry.all()):
                from icrl.goal_reaching_evaluation import evaluate_goal_reaching_policy
                from stable_baselines3.common.vec_env import DummyVecEnv

                env = DummyVecEnv([lambda: gym.make('KukaCup1-v0')])
                try:
                    policy = DS_Policy(3, lambda obs: 2 * obs[3:6], stable_modulation_basis=True)
                    result = evaluate_goal_reaching_policy(
                        policy, env, env.get_attr('true_cost_function')[0], 1)
                    assert result['goal_reached_portion'] == 1
                finally:
                    env.close()

            for env_id in ('ReachObs-v0', 'ReachVel-v0', 'ReachConcaveObs-v0'):
                env = gym.make(env_id)
                try:
                    env.reset()
                    env.step(env.action_space.sample())
                finally:
                    env.close()
            assert not sitecustomize.attempts, sitecustomize.attempts
        ''')

    def test_mujoco_is_required_only_when_its_environment_is_requested(self):
        self.run_isolated('''
            import sitecustomize
            import gym
            import custom_envs
            import custom_envs.envs as envs
            assert not sitecustomize.attempts
            try:
                gym.make('HalfCheetahTest-v0')
            except (ImportError, gym.error.DependencyNotInstalled):
                pass
            else:
                raise AssertionError('MuJoCo environment ran with its dependency blocked')
            assert sitecustomize.attempts

            # Preserve the legacy package-level class API, with lazy loading.
            try:
                envs.AntTest
            except (ImportError, gym.error.DependencyNotInstalled):
                pass
            else:
                raise AssertionError('MuJoCo class access did not load its implementation')
        ''')


if __name__ == '__main__':
    unittest.main()
