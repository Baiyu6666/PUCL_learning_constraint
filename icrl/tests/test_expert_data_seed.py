import pickle
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from icrl.utils import load_expert_data


class ExpertDataSeedTests(unittest.TestCase):
    def test_selection_is_repeatable_and_preserves_training_rngs(self):
        with tempfile.TemporaryDirectory() as directory:
            rollouts = Path(directory) / 'files/EXPERT/rollouts'
            rollouts.mkdir(parents=True)
            for index in range(10):
                with (rollouts / f'{index}.pkl').open('wb') as stream:
                    pickle.dump(dict(observations=np.array([[index]]),
                                     actions=np.zeros((1, 1)),
                                     rewards=np.array([0.]), lengths=1), stream)
            for seed in (None, 0, 23):
                with self.subTest(seed=seed):
                    random.seed(7)
                    np.random.seed(7)
                    torch.manual_seed(7)
                    expected = (random.random(), np.random.rand(), torch.rand(3))
                    random.seed(7)
                    np.random.seed(7)
                    torch.manual_seed(7)
                    first = load_expert_data(directory, 4, selection_seed=seed)
                    self.assertEqual(random.random(), expected[0])
                    self.assertEqual(np.random.rand(), expected[1])
                    self.assertTrue(torch.equal(torch.rand(3), expected[2]))
                    second = load_expert_data(directory, 4, selection_seed=seed)
                    np.testing.assert_array_equal(first[0][0], second[0][0])
                    ids = np.random.RandomState(0 if seed is None else seed).choice(10, 4, replace=False)
                    np.testing.assert_array_equal(first[0][0].ravel(), ids)


if __name__ == '__main__':
    unittest.main()
