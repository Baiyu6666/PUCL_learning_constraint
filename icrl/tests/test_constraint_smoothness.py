import unittest

import torch
from torch import nn

from icrl.constraint_net import ConstraintNet


class SmoothnessTests(unittest.TestCase):
    def model(self):
        return ConstraintNet(3, 0, (32, 32), 64, lambda _: 0.005, False,
                             obs_select_dim=[0, 1, 2], acs_select_dim=[-1])

    def test_linear_prediction_has_zero_penalty(self):
        model = self.model()
        model.network = nn.Linear(3, 1).double()
        inputs = torch.randn(64, 3, dtype=torch.float64)
        self.assertLess(model.local_curvature_loss(inputs, 0.01).item(), 1e-20)

    def test_penalty_backpropagates_to_network(self):
        model = self.model()
        inputs = torch.rand(64, 3)
        loss = model.local_curvature_loss(inputs, 0.01)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_coordinate_rotation_does_not_change_penalty(self):
        model = self.model()
        inputs = torch.randn(64, 3, dtype=torch.float64)
        directions = torch.randn_like(inputs)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))

        class Quadratic(nn.Module):
            def __init__(self, transform):
                super().__init__()
                self.transform = transform

            def forward(self, x):
                return ((x @ self.transform) ** 2 * x.new_tensor([1, 2, 3])).sum(1, keepdim=True)

        model.network = Quadratic(torch.eye(3, dtype=torch.float64))
        original = model.local_curvature_loss(inputs, 0.01, directions=directions)
        model.network = Quadratic(rotation.T)
        rotated = model.local_curvature_loss(inputs @ rotation, 0.01, directions=directions @ rotation)
        torch.testing.assert_close(original, rotated, rtol=1e-8, atol=1e-10)

    def test_private_generator_does_not_change_global_rng(self):
        model = self.model()
        inputs = torch.ones(64, 3)
        generator = torch.Generator().manual_seed(9173)
        state = torch.random.get_rng_state().clone()
        model.local_curvature_loss(inputs, 0.01, generator=generator)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
