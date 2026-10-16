import unittest
from types import SimpleNamespace
import numpy as np
import torch
from icrl.dscl import dscl, resolve_cn_smoothness_coeff, resolve_select_policy_excerpts
from icrl.constraint_net import ConstraintNet


class MECLBaselineTests(unittest.TestCase):
    def test_defaults_preserve_pucl_and_do_not_enable_pucl_components_for_mecl(self):
        self.assertEqual(resolve_cn_smoothness_coeff('KukaCup1-v0',None),.01)
        self.assertTrue(resolve_select_policy_excerpts('KukaCup1-v0',None))
        self.assertEqual(resolve_cn_smoothness_coeff('KukaCup1-v0',None,'mecl'),0.)
        self.assertFalse(resolve_select_policy_excerpts('KukaCup1-v0',None,'mecl'))

    def test_mecl_rejects_unsupported_pucl_smoothness_before_creating_envs(self):
        config=SimpleNamespace(use_wandb=False,constraint_learning_method='mecl',
            train_env_id='KukaCup1-v0',cn_smoothness_coeff=.01,select_policy_excerpts=None,
            train_constraint_with_true_data=False)
        with self.assertRaisesRegex(ValueError,'no PUCL smoothness'):
            dscl(config)

    def test_ml_loss_moves_expert_and_nominal_predictions_in_correct_directions(self):
        torch.manual_seed(23);torch.set_num_threads(1)
        model=ConstraintNet(3,3,[],64,lambda _: .02,False,.01,[0,1,2],[-1],
            no_importance_sampling=True,loss_type='ml',manual_threshold=.5,device='cpu')
        with torch.no_grad():
            for parameter in model.network.parameters():parameter.zero_()
        model.optimizer=torch.optim.SGD(model.network.parameters(),lr=.1)
        experts=np.tile([1.,0.,0.],(64,1));nominal=-experts;actions=np.zeros_like(experts)
        metrics=model.train_MECL_BC(1,experts,actions,nominal,actions,np.array([64]),1.)
        self.assertTrue(np.isfinite(metrics['backward/cn_loss']))
        self.assertLess(model.cost_function_non_binary(experts,actions).mean(),.5)
        self.assertGreater(model.cost_function_non_binary(nominal,actions).mean(),.5)


if __name__=='__main__':unittest.main()
