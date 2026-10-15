import unittest

import numpy as np

from icrl.ds_policy import DS_Policy


class ReferenceModulationTests(unittest.TestCase):
    def test_xy_reference_follows_state_height_for_horizontal_normal(self):
        policy = DS_Policy(3, lambda x: np.array([1., 1., 1.]), stable_modulation_basis=True)
        policy.modulate_with_NN_with_refer_point(
            lambda obs, acs: (np.array([2.]), np.array([1., 0., 0.])),
            refer_point=[0, 0], rho=1)
        observations = np.array([[1., 1., 0.2], [1., 1., 0.8]])
        velocity, _ = policy.predict(observations, velocity_normalization=False)
        # E D E^-1 with reference [-1,-1,0] and identical tangent eigenvalues.
        np.testing.assert_allclose(velocity, [[.5, .5, 1.5], [.5, .5, 1.5]])

    def test_reference_modulation_uses_requested_rho(self):
        velocities = []
        for rho in (.2, 2):
            policy = DS_Policy(3, lambda x: np.array([1., 0., 0.]), stable_modulation_basis=True)
            policy.modulate_with_NN_with_refer_point(
                lambda obs, acs: (np.array([4.]), np.array([1., 0., 0.])), refer_point=[0, 0], rho=rho)
            velocity, _ = policy.predict(np.array([[1., 0., .3]]), velocity_normalization=False)
            np.testing.assert_allclose(velocity[0, 0], 1-4**(-1/rho))
            velocities.append(velocity[0, 0])
        self.assertGreater(velocities[0], velocities[1])

    def test_stable_basis_matches_original_reference_matrix_when_nonsingular(self):
        reference = np.array([.60, -.06])
        state = np.array([.75, .12, .4])
        normal = np.array([1., 2., 3.])
        refer = np.r_[reference, state[2]] - state
        original_basis = np.stack([refer, [normal[1], -normal[0], 0],
                                   [0, normal[2], -normal[1]]], axis=1)
        expected = original_basis @ np.diag([.5, 1.5, 1.5]) @ np.linalg.inv(original_basis) @ np.ones(3)
        policy = DS_Policy(3, lambda x: np.ones(3), stable_modulation_basis=True)
        policy.modulate_with_NN_with_refer_point(
            lambda obs, acs: (np.array([2.]), normal), refer_point=reference, rho=1)
        actual, _ = policy.predict(state[None], velocity_normalization=False)
        np.testing.assert_allclose(actual[0], expected)

    def test_singular_reference_basis_falls_back_to_normal(self):
        for point in ([1., 0.], [0., 0.]):
            policy = DS_Policy(3, lambda x: np.ones(3), stable_modulation_basis=True)
            policy.modulate_with_NN_with_refer_point(
                lambda obs, acs: (np.array([2.]), np.array([0., 1., 0.])), refer_point=point, rho=1)
            velocity, _ = policy.predict(np.array([[1., 0., .3]]), velocity_normalization=False)
            np.testing.assert_allclose(velocity, [[1.5, .5, 1.5]])

    def test_legacy_concave_scalar_argument_still_uses_original_reference(self):
        gamma = lambda obs, acs: (np.array([2.]), np.array([1., 2., 3.]))
        velocities = []
        for reference in (None, 2.0):
            policy = DS_Policy(3, lambda x: np.ones(3))
            policy.modulate_with_NN_with_refer_point(gamma, refer_point=reference)
            velocities.append(policy.predict(np.array([[.7, .2, .3]]), velocity_normalization=False)[0])
        np.testing.assert_array_equal(*velocities)


if __name__ == '__main__':
    unittest.main()
