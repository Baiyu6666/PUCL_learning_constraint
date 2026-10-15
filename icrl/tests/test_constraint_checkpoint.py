import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from icrl.constraint_net import ConstraintNet


class ConstraintCheckpointTests(unittest.TestCase):
    def test_threshold_round_trip_and_legacy_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'models/icrl_1_itrs/cn.pt'
            path.parent.mkdir(parents=True)
            model = ConstraintNet(3, 1, [4], None, None, is_discrete=False,
                                  optimizer_class=None, obs_select_dim=[0, 1, 2],
                                  acs_select_dim=[-1], manual_threshold=0.63)
            model.save(path)
            (root / 'config.json').write_text(json.dumps({'cn_manual_threshold': 0.41}))
            loaded = ConstraintNet.load(path)
            self.assertEqual(loaded.manual_threshold, 0.63)
            obs, acs = np.ones((2, 3)), np.zeros((2, 1))
            np.testing.assert_array_equal(model.cost_function(obs, acs), loaded.cost_function(obs, acs))
            np.testing.assert_array_equal(model.build_gamma_with_grad_for_ds(obs, acs)[0],
                                          loaded.build_gamma_with_grad_for_ds(obs, acs)[0])
            self.assertEqual(ConstraintNet.load(path, manual_threshold=0.52).manual_threshold, 0.52)
            state = torch.load(path)
            del state['manual_threshold']
            torch.save(state, path)
            self.assertEqual(ConstraintNet.load(path).manual_threshold, 0.41)
            model._load(path)
            self.assertEqual(model.manual_threshold, 0.41)
            (root / 'config.json').unlink()
            with self.assertWarnsRegex(UserWarning, 'using 0.5'):
                self.assertEqual(ConstraintNet.load(path).manual_threshold, 0.5)


if __name__ == '__main__':
    unittest.main()
