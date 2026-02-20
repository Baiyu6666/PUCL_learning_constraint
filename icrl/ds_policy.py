import numpy as np
from typing import Any, Callable, Optional, Tuple
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from icrl.true_constraint_net import get_true_cost_function
from icrl.utils import make_eval_env, del_and_make, save_dict_as_pkl, sample_from_agent
from icrl.constraint_net import plot_constraints, ConstraintNet
from icrl.utils import load_expert_data, sample_trajectories_for_reach_env
import matplotlib

matplotlib.use('TKAgg')

class DS_Policy():
    def __init__(self, obs_dim, nominal_ds, gmm_confidence=5):
        # nominal_ds is a numpy function x:n-d np array and u:m-d numpy array into n=d np array
        self.nominal_ds = nominal_ds
        self.modulated_ds = nominal_ds
        self.obs_dim = obs_dim
        self.gmm_confidence = gmm_confidence

    def __call__(self, *args):
        return self.predict(*args)

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[np.ndarray] = None,
            deterministic: bool = True,
            velocity_normalization = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Note: self.modulated ds accept only 1-D array
        x_dot = np.apply_along_axis(self.modulated_ds, 1, observation)
        #TODO: formally decide velocity magnitude after modulation
        if velocity_normalization:
            x_dot = np.apply_along_axis(lambda v: v / np.linalg.norm(v) * 1. if np.linalg.norm(v) > 0.05 else v, 1, x_dot)
        return x_dot, state

    def modulate_with_gmm_constraint_2d(self, gmm, rho=0.05, yita=1.1):  # rho=0.05, yita=1.05
        def compute_M(obs):
            M = np.eye(2)
            for i in range(gmm.n_components):
                x, y = (obs[0] - gmm.means_[i, 0])/yita, (obs[1] - gmm.means_[i, 1])/yita
                obs_centered = np.array([[x], [y]])
                dis2center = (x**2+y**2)**0.5
                cov = gmm.covariances_[i]
                cov_inv = np.linalg.inv(cov)
                gamma = 1 + max(dis2center - dis2center * self.gmm_confidence / (obs_centered.T @ cov_inv @ obs_centered).item(), 0) #*2
                gamma = gamma**(1/rho)
                normal = np.array([[2 * x * cov_inv[0, 0] + 2 * y * cov_inv[0, 1]], [2 * y * cov_inv[1, 1] + 2 * x * cov_inv[0, 1]]])

                refer = obs_centered
                tangent = np.array([[normal[1,0]], [-normal[0,0]]])
                E = np.concatenate([refer, tangent], 1)
                E = E / np.linalg.norm(E, axis=0)
                D = np.diag([1-1/gamma, 1+1/gamma])
                M = E @ D @ np.linalg.inv(E) @ M
            return M, [E, D, gamma]
        self.modulated_ds = lambda x: compute_M(x)[0] @ self.nominal_ds(x)
        # self.get_gamma = lambda x: compute_M(x)[1][2]
        self.gmm = gmm
        # self.modulated_ds = lambda x: modulated_ds(x) / np.linalg.norm(modulated_ds(x)) * np.linalg.norm(x)
        # TODO: modulation of multiple obstacle is not trivial and need further investigation
        # TODO: consider tail effect

    def modulate_with_gmm_constraint_3d(self, gmm, rho=0.05, yita=1.):  # rho=0.05, yita=1.05
        def compute_M(obs):
            M = np.eye(3)
            for i in range(gmm.n_components):
                x, y = (obs[0] - gmm.means_[i, 0])/yita, (obs[1] - gmm.means_[i, 1])/yita

                obs_centered = np.array([[x], [y]])
                dis2center = (x**2+y**2)**0.5
                cov = gmm.covariances_[i]
                cov_inv = np.linalg.inv(cov)
                gamma = 1 + max(dis2center - dis2center * self.gmm_confidence / (obs_centered.T @ cov_inv @ obs_centered).item(), 0) #*2
                gamma = gamma**(1/rho)
                normal = np.array([[2 * x * cov_inv[0, 0] + 2 * y * cov_inv[0, 1]], [2 * y * cov_inv[1, 1] + 2 * x * cov_inv[0, 1]]])

                refer = np.array([[x], [y], [obs[2]]])
                tangent1 = np.array([normal[1, 0], -normal[0, 0], 0])[:,np.newaxis]
                tangent2 = np.array([0, 0, -normal[1, 0]])[:,np.newaxis]
                E = np.concatenate([refer, tangent1, tangent2], 1)
                D = np.diag([1 - 1 / gamma, 1 + 1 / gamma, 1 + 1 / gamma])
                if (np.linalg.norm(E, axis=0) < 1e-5).any():
                    # print(f'Warning! Too small E encountered, E={np.linalg.norm(E, axis=0)}, gamma={gamma}')
                    return np.eye(3), E, np.eye(3), 10
                E = E / np.linalg.norm(E, axis=0)
                M = E @ D @ np.linalg.inv(E) @ M
            return M, [E, D, gamma]
        self.modulated_ds = lambda x: compute_M(x)[0] @ self.nominal_ds(x)
        # self.get_gamma = lambda x: compute_M(x)[1][2]
        self.gmm = gmm


    def modulate_with_NN(self, gamma_function, rho=0.01, yita=1):  # rho=0.05, yita=1.05
        def compute_M(obs):
            action = np.zeros(self.obs_dim)
            gamma, normal = gamma_function(obs, action)
            gamma = gamma.item()
            gamma = gamma ** (1 / rho)
            assert gamma >= 1
            cn_obs_dim = normal.shape[0]
            if cn_obs_dim == 2:
                tangent = np.array([normal[1], -normal[0]])
                E = np.stack([normal, tangent], -1)
                D = np.diag([1 - 1 / gamma, 1 + 1 / gamma])
            elif cn_obs_dim == 3:
                tangent1 = np.array([normal[1], -normal[0], 0])
                tangent2 = np.array([0, normal[2], -normal[1]])
                E = np.stack([normal, tangent1, tangent2], -1)
                D = np.diag([1 - 1 / gamma, 1 + 1 / gamma, 1 + 1 / gamma])
            else:
                raise ValueError('Unsupported dimension of obs vector')

            if (np.linalg.norm(E, axis=0) < 1e-5).any():
                # print(f'Warning! Too small E encountered, E={np.linalg.norm(E, axis=0)}, gamma={gamma}')
                return np.eye(cn_obs_dim), E, np.eye(cn_obs_dim), 10
            E = E / np.linalg.norm(E, axis=0)
            M = E @ D @ np.linalg.inv(E)
            return M, E, D, gamma
        self.modulated_ds = lambda x: compute_M(x)[0] @ self.nominal_ds(x)

    def modulate_with_NN_with_refer_point(self, gamma_function, refer_point=None, rho=1, yita=1):  # rho=0.05, yita=1.05
        def compute_M(obs):
            action = np.zeros(self.obs_dim)
            gamma, normal = gamma_function(obs, action)
            gamma = gamma.item()
            gamma = gamma ** (1 / rho)

            assert gamma >= 1
            cn_obs_dim = normal.shape[0]
            if cn_obs_dim == 2:
                refer = refer_point - obs[:2]
                tangent = np.array([normal[1], -normal[0]])
                E = np.stack([refer, tangent], -1)
                D = np.diag([1 - 1 / gamma, 1 + 1 / gamma])
            elif cn_obs_dim == 3:
                refer = np.array([0.55, 0, obs[2]]) - obs[:3] # for panda concave refer point is fixed
                tangent1 = np.array([normal[1], -normal[0], 0])
                tangent2 = np.array([0, normal[2], -normal[1]])
                E = np.stack([refer, tangent1, tangent2], -1)
                D = np.diag([1 - 1 / gamma, 1 + 1 / gamma, 1 + 1 / gamma])
            else:
                raise ValueError('Unsupported dimension of obs vector')
            if (np.linalg.norm(E, axis=0) < 1e-5).any():
                return np.eye(cn_obs_dim), E, np.eye(cn_obs_dim), 10
            E = E / np.linalg.norm(E, axis=0)
            M = E @ D @ np.linalg.inv(E)
            return M, E, D, gamma
        self.modulated_ds = lambda x: compute_M(x)[0] @ self.nominal_ds(x)

    def modulate_with_cylinder_3d(self, rho=0.1, yita=1):  # rho=0.05, yita=1.05
        # This function modulate the nominal DS with the ground truth cylinder constraint
        def compute_M(obs):
            refer_point = np.array([0.55, 0, obs[2]])
            refer = (refer_point - obs[:3]).reshape(-1, 1)
            pos_distance = max(((obs[0] - 0.55) ** 2 + (obs[1]) ** 2 - 0.05 ** 2), 0)
            gamma = 1 + np.sqrt(np.abs(pos_distance))
            gamma = gamma ** (1 / rho)
            normal = np.array([[obs[0] - 0.55], [obs[1]], [0]])
            tangent1 = np.array([[normal[1, 0]], [-normal[0, 0]], [0]])
            tangent2 = np.array([[0], [normal[2, 0]], [-normal[1, 0]]])
            E = np.concatenate([refer, tangent1, tangent2], 1)
            E = E / np.linalg.norm(E, axis=0)
            D = np.diag([1 - 1 / gamma, 1 + 1 / gamma, 1 + 1 / gamma])
            M = E @ D @ np.linalg.inv(E)
            return M, [E, D, gamma]
        self.modulated_ds = lambda x: compute_M(x)[0] @ self.nominal_ds(x) * 3

    def generate_demonstrations_PointObs(self, save_data=False, generate_failed_demo=False, n_rollouts=100):
        env = make_eval_env('PointDSTest-v0', use_cost_wrapper=False, normalize_obs=False)
        if generate_failed_demo:
            save_dir = 'icrl/expert_data/PointDS/files/FAILED/rollouts'
        else:
            save_dir = 'icrl/expert_data/PointDS/files/EXPERT/rollouts'
        del_and_make(save_dir) if save_data else None

        gmm = env.get_attr('gmm_true')[0]
        if generate_failed_demo == False:
            self.modulate_with_gmm_constraint_2d(gmm)

        x_start, x_end = 0.4, 1.7
        y_start, y_end = 0.4, 1.7

        x = np.arange(x_start, x_end, 0.25)
        y = np.arange(y_start, y_end, 0.25)
        x, y = np.meshgrid(x, y)
        starting_points = np.vstack([x.ravel(), y.ravel()]).T

        obs_all, len_all = [], []
        idx = 0
        for starting_point in starting_points:
            # Only use feasible starting points
            if env.env_method('true_cost_function', starting_point, acs=None)[0] == 0:
                saving_dict = sample_from_agent(self, env, 1, False, starting_point, return_feasibility=True)
                observations, _, actions, rewards, lengths, feasibles = saving_dict
                # Save failed demonstrations only if specified.
                # if (failed_demo and feasibles[0] == 1) or (failed_demo==False and feasibles[0]==0):
                #     continue
                saving_dict = dict(observations=observations, actions=actions, rewards=rewards, lengths=lengths)
                print(f'Generate one trajectory with length {lengths[0]} and reward {rewards[0]}')
                obs_all.append(observations)
                len_all.append(int(lengths))
                saving_dict['save_scheme'] = 'not_airl'
                save_dict_as_pkl(saving_dict, save_dir, str(idx)) if save_data else None
                idx += 1
                if idx == n_rollouts:
                    break
        obs_all = np.vstack(obs_all)
        fig, ax = plt.subplots(1, 1, figsize=(13, 13))
        start = 0
        for i, length in enumerate(len_all):
            ax.scatter(obs_all[np.arange(start, start + length, 1), 0],
                    obs_all[np.arange(start, start + length, 1), 1], clip_on=False, linewidth=3)
            start += length
        plot_gmm_ellipse(gmm, ax, edgecolor='k')
        self.plot_vector_flow(ax)
        ax.tick_params(labelsize=20)
        plt.grid('on')
        plt.title('Demonstration', fontsize=30)
        fig.savefig(save_dir + 'demonstration.png', bbox_inches='tight')
        print(f'Successfully generate {idx} demonstrations')

        plt.show()
        plt.close(fig=fig)

    def generate_demonstrations_ReachObs(self, generate_failed_demo=False, n_rollouts=100):
        env = make_eval_env('ReachObs-v0', use_cost_wrapper=False, normalize_obs=False)
        if generate_failed_demo:
            save_dir = 'icrl/expert_data/ReachObsDS/files/FAILED/rollouts'
        else:
            save_dir = 'icrl/expert_data/ReachObsDS/files/EXPERT/rollouts'
        del_and_make(save_dir)

        if generate_failed_demo == False:
            self.modulate_with_cylinder_3d()

        x_start, x_end = 0.35, 0.48
        y_start, y_end = -0.09, 0.09
        z_start, z_end = 0.05, 0.1

        x = np.arange(x_start, x_end, 0.09)
        y = np.arange(y_start, y_end, 0.06)
        z = np.arange(z_start, z_end, 0.04)
        x, y, z = np.meshgrid(x, y, z)
        starting_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

        obs_all, len_all = [], []
        idx = 0
        for starting_point in starting_points:
            # Only use feasible starting points
            if env.env_method('true_cost_function', starting_point, acs=None)[0] == 0:
                saving_dict = sample_from_agent(self, env, 1, False, starting_point, return_feasibility=True)
                observations, _, actions, rewards, lengths, feasibles = saving_dict
                # Only save successful demonstrations or failed demonstrations if specified.
                if (generate_failed_demo and feasibles[0] == 1) or (generate_failed_demo == False and feasibles[0] == 0):
                    continue
                # Drop infeasible trajectories
                saving_dict = dict(observations=observations, actions=actions, rewards=rewards, lengths=lengths)
                print(f'Generate one trajectory with length {lengths[0]} and reward {rewards[0]}')
                obs_all.append(observations)
                len_all.append(int(lengths))
                saving_dict['save_scheme'] = 'not_airl'
                # save_dict_as_pkl(saving_dict, save_dir, str(idx))
                idx += 1
                if idx == n_rollouts:
                    break

        plot_constraints(cost_function=None, manual_threshold=0.5, position_limit=None, env=env, env_id='ReachObs-v0',
                         select_dim=[0, 1, 2], obs_dim=9, acs_dim=3, save_name=save_dir + 'demonstration.png',
                         fig_title='demonstration', obs_expert=np.vstack(obs_all), rew_expert=None, len_expert=len_all,
                         obs_nominal=None, rew_nominal=None, len_nominal=None, policy_excerpt_rollout_num=None,
                         obs_failed=None, obs_rn=None, gmm_learned=None, query_obs=None, show=not False)

    def generate_demonstrations_ReachConcaveObs(self, generate_failed_demo=False, n_rollouts=100):
        if generate_failed_demo:
            save_dir = 'icrl/expert_data/ReachConcaveObsDS/files/FAILED/rollouts'
        else:
            save_dir = 'icrl/expert_data/ReachConcaveObsDS/files/EXPERT/rollouts'
        del_and_make(save_dir)
        env = make_eval_env('ReachConcaveObs-v0', use_cost_wrapper=False, normalize_obs=False)
        gmm = env.get_attr('gmm_true')[0]

        # Load the constraint net trained on ground truth data
        # constraint_net = ConstraintNet.load(f'icrl/wandb/run-20240715_162340-jq57sx5y/files/models/icrl_1_itrs/cn.pt')
        constraint_net = ConstraintNet.load(f'icrl/wandb/run-20240715_181551-z19huxv4/files/models/icrl_1_itrs/cn.pt')

        if generate_failed_demo == False:
            self.modulate_with_NN__with_refer_point(constraint_net.build_gamma_with_grad_for_ds, rho=2)
            # self.modulate_with_gmm_constraint_3d(gmm, rho=0.05)

        x_start, x_end = 0.4, 0.47
        y_start, y_end = -0.1, 0.1
        z_start, z_end = 0.05, 0.36

        x = np.linspace(x_start, x_end, 3)
        y = np.linspace(y_start, y_end, 3)
        z = np.linspace(z_start, z_end, 3)
        x, y, z = np.meshgrid(x, y, z)
        # Set up starting points that generate meaningful trajectories
        starting_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
        starting_points = np.concatenate(
            (np.array([[ 0.46  ,  0.032 ,  0.125 ],
                       [ 0.404 , -0.0608,  0.26  ],
                       [ 0.46  , +0.0096,  0.36  ],
                       [ 0.398 ,  0.128 ,  0.24  ],
                       [ 0.37  ,  0.064 ,  0.34  ],
                       [ 0.55  ,  0.1   ,  0.3   ],
                       [ 0.6   , -0.0148,  0.14  ],
                       [ 0.42  , -0.0148,  0.075 ],
                       [ 0.43  , -0.    ,  0.185 ],
                       [ 0.59  ,  0.016 ,  0.05  ],
                       [0.59, 0.016, 0.2],
                       [ 0.558 , -0.016 ,  0.065 ]]), starting_points))

        # Delete points that are too close to each other
        selected_points = []
        for point in starting_points:
            if all(np.linalg.norm(point - p) > 0.05 for p in selected_points):
                selected_points.append(point)
        starting_points = np.array(selected_points)

        obs_all, len_all = [], []
        idx = 0
        for starting_point in starting_points:
            # Only use feasible starting points
            if env.env_method('true_cost_function', starting_point, acs=None)[0] == 0:
                saving_dict = sample_from_agent(self, env, 1, False, starting_point, return_feasibility=True)
                observations, _, actions, rewards, lengths, feasibles = saving_dict
                # Only save successful demonstrations or failed demonstrations if specified.
                if (generate_failed_demo and feasibles[0] == 1) or (generate_failed_demo == False and feasibles[0] == 0) or lengths > 300:
                    print('Generate one invaliad trajectory and discard')
                    continue
                # Drop infeasible trajectories
                saving_dict = dict(observations=observations, actions=actions, rewards=rewards, lengths=lengths)
                print(f'Generate one trajectory with length {lengths[0]} and reward {rewards[0]}')
                obs_all.append(observations)
                len_all.append(int(lengths))
                saving_dict['save_scheme'] = 'not_airl'
                save_dict_as_pkl(saving_dict, save_dir, str(idx))
                idx += 1
                if idx == n_rollouts:
                    break

        plot_constraints(cost_function=constraint_net.cost_function, manual_threshold=0.5, position_limit=None, env=env,
                         env_id='ReachObs-v0', select_dim=[0, 1, 2], obs_dim=9, acs_dim=3,
                         save_name=save_dir + 'demonstration.png', fig_title='demonstration',
                         obs_expert=np.vstack(obs_all), rew_expert=None, len_expert=len_all, obs_nominal=None,
                         rew_nominal=None, len_nominal=None, policy_excerpt_rollout_num=None, obs_failed=None,
                         obs_rn=None, gmm_learned=None, query_obs=None, show=not False, num_points_for_plot=30)

    def generate_demonstrations_Reach2RegionObs(self, gnerate_failed_demo=False, n_rollouts=100):
        if gnerate_failed_demo:
            save_dir = 'icrl/expert_data/Reach2RegionObsDS/files/FAILED/rollouts'
        else:
            save_dir = 'icrl/expert_data/Reach2RegionObsDS/files/EXPERT/rollouts'
        del_and_make(save_dir)
        env = make_eval_env('Reach2RegionObs-v0', use_cost_wrapper=False, normalize_obs=False)
        gmm = env.get_attr('gmm_true')[0]

        # constraint_net = ConstraintNet.load(f'icrl/wandb/run-20240715_162340-jq57sx5y/files/models/icrl_1_itrs/cn.pt')
        constraint_net = ConstraintNet.load(f'icrl/wandb/run-20240717_234544-biivmhhm/files/models/icrl_1_itrs/cn.pt')

        if gnerate_failed_demo == False:
            self.modulate_with_NN(constraint_net.build_gamma_with_grad_for_ds, rho=2)
            # self.modulate_with_gmm_constraint_3d(gmm, rho=0.05)

        x_start, x_end = 0.40, 0.47
        y_start, y_end = -0.15, 0.2
        z_start, z_end = 0.05, 0.36

        x = np.linspace(x_start, x_end, 1)
        y = np.linspace(y_start, y_end, 5)
        z = np.linspace(z_start, z_end, 3)
        x, y, z = np.meshgrid(x, y, z)
        starting_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
        starting_points = np.concatenate(
            (np.array([[0.6, -0.03, 0.1],
                       [0.6, -0.03, 0.25],
                       [0.6, -0.03, 0.17],
                       [0.57, 0.05, 0.1],
                       [0.57, 0.05, 0.17],
                       [0.57, 0.05, 0.25],
                       [0.47, -0.15, 0.17],
                       [0.47, -0.15, 0.1],
                       [0.4, 0.2, 0.25]

                       ]), starting_points))

        selected_points = []
        for point in starting_points:
            if all(np.linalg.norm(point - p) > 0.05 for p in selected_points):
                selected_points.append(point)
        starting_points = np.array(selected_points)

        obs_all, len_all = [], []
        idx = 0
        for starting_point in starting_points:
            # Only use feasible starting points
            if env.env_method('true_cost_function', starting_point, acs=None)[0] == 0:
                saving_dict = sample_from_agent(self, env, 1, False, starting_point, return_feasibility=True)
                observations, _, actions, rewards, lengths, feasibles = saving_dict
                # Drop infeasible trajectories
                # if (failed_demo and feasibles[0] == 1) or (failed_demo == False and feasibles[0] == 0) or lengths > 300:
                #     print('Generate one failed trajectory and discard')
                #     continue
                saving_dict = dict(observations=observations, actions=actions, rewards=rewards, lengths=lengths)
                print(f'Generate one trajectory with length {lengths[0]} and reward {rewards[0]}')
                obs_all.append(observations)
                len_all.append(int(lengths))
                saving_dict['save_scheme'] = 'not_airl'
                save_dict_as_pkl(saving_dict, save_dir, str(idx))
                idx += 1
                if idx == n_rollouts:
                    break

        plot_constraints(cost_function=constraint_net.cost_function, manual_threshold=0.5, position_limit=None, env=env,
                         env_id='ReachObs-v0', select_dim=[0, 1, 2], obs_dim=9, acs_dim=3,
                         save_name=save_dir + 'demonstration.png', fig_title='demonstration',
                         obs_expert=np.vstack(obs_all), rew_expert=None, len_expert=len_all, obs_nominal=None,
                         rew_nominal=None, len_nominal=None, policy_excerpt_rollout_num=None, obs_failed=None,
                         obs_rn=None, gmm_learned=None, query_obs=None, show=not False, num_points_for_plot=30)

    def generate_demonstrations_point_ellip(self, failed_demo=False, n_rollouts=100):
        if failed_demo:
            save_dir = 'icrl/expert_data/PointEllip/files/FAILED/rollouts'
        else:
            save_dir = 'icrl/expert_data/PointEllip/files/EXPERT/rollouts'
        del_and_make(save_dir)
        env = make_eval_env('PointEllipTest-v0', use_cost_wrapper=False, normalize_obs=False)
        gmm = env.get_attr('gmm_true')[0]
        if failed_demo == False:
            self.modulate_with_gmm_constraint_2d(gmm)

        x_start, x_end = 0.43, 1.4
        y_start, y_end = 0.43, 1.6

        x = np.arange(x_start, x_end, 0.2)
        y = np.arange(y_start, y_end, 0.2)
        x, y = np.meshgrid(x, y)
        starting_points = np.vstack([x.ravel(), y.ravel()]).T

        obs_all, len_all = [], []
        idx = 0
        for starting_point in starting_points:
            # Only use feasible starting points
            if env.env_method('true_cost_function', starting_point, acs=None)[0] == 0:
                saving_dict = sample_from_agent(self, env, 1, False, starting_point, return_feasibility=True)
                observations, _, actions, rewards, lengths, feasibles = saving_dict
                # Only save successful demonstrations or failed demonstrations if specified.
                if (failed_demo and feasibles[0] == 1) or (failed_demo == False and feasibles[0] == 0):
                    continue
                # Drop infeasible trajectories
                saving_dict = dict(observations=observations, actions=actions, rewards=rewards, lengths=lengths)
                print(f'Generate one trajectory with length {lengths[0]} and reward {rewards[0]}')
                obs_all.append(observations)
                len_all.append(int(lengths))
                saving_dict['save_scheme'] = 'not_airl'
                # save_dict_as_pkl(saving_dict, save_dir, str(idx))
                idx += 1
                if idx == n_rollouts:
                    break
        obs_all = np.vstack(obs_all)

        import matplotlib
        matplotlib.use('Agg')
        fig, ax = plt.subplots(1, 1, figsize=(13, 13))
        start = 0
        for i, length in enumerate(len_all):
            ax.plot(obs_all[np.arange(start, start + length, 1), 0],
                    obs_all[np.arange(start, start + length, 1), 1], clip_on=False, linewidth=3)
            start += length
        plot_gmm_ellipse(gmm, ax, edgecolor='k')
        self.plot_vector_flow(ax)
        ax.tick_params(labelsize=20)
        plt.grid('on')
        plt.title('Demonstration', fontsize=30)
        fig.savefig(save_dir + 'demonstration.png', bbox_inches='tight')
        print(f'Successfully generate {idx} demonstrations')

        plt.show()
        plt.close(fig=fig)

    def plot_vector_flow(self, ax, gap=0.07):
        x_start, x_end = -0.5, 2.
        y_start, y_end = -0.5, 2.

        x = np.arange(x_start, x_end + 0.1, gap)
        y = np.arange(y_start, y_end + 0.1, gap)
        x, y = np.meshgrid(x, y)
        x_eval = np.vstack([x.ravel(), y.ravel()]).T
        fx_eval, _ = self.predict(x_eval)
        ax.quiver(x_eval[:, 0], x_eval[:, 1], fx_eval[:, 0], fx_eval[:, 1], color='white', scale=50,
                   label='Velocity Vectors')
        # for x_eval_point in x_eval:
        #     E = self.E(x_eval_point)
        #     gamma = self.gamma(x_eval_point)
        #     plt.quiver(x_eval_point[0], x_eval_point[1], E[0, 0], E[0, 1], color='gray', scale=40)
        #     ax.text(x_eval_point[0], x_eval_point[1], f'{gamma:.2f}', ha='left', va='bottom')


def plot_gmm_ellipse(gmm, ax, **kwargs):
    for means, covariances in zip(gmm.means_, gmm.covariances_):
        vals, vecs = np.linalg.eigh(covariances)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals * gmm.confidence)
        ellipse = Ellipse(xy=means, width=width, height=height, angle=theta, fc='None', lw=3, **kwargs)
        ax.add_patch(ellipse)


def generate_policy_with_DS_and_random_goal(env_id, load_cn_dir, load_cn_itr, n_goals=8, n_rollouts=15):  # safe_demo, failed_demo, policy

    save_dir = f'icrl/wandb/{load_cn_dir}/files/transfer/rollouts'
    del_and_make(save_dir)

    if 'ReachConcave' in env_id:
        A = np.diag([-2, -2, -2])
        x_start, x_end = 0.55, 0.65
        y_start, y_end = -0.16, 0.16
        z_start, z_end = 0.05, 0.25
    elif 'Reach2Region' in env_id:
        A = np.diag([-2, -2, -2])
        x_start, x_end = 0.55, 0.65
        y_start, y_end = -0.16, 0.16
        z_start, z_end = 0.05, 0.18

    constraint_net = ConstraintNet.load(f'icrl/wandb/{load_cn_dir}/files/models/icrl_{load_cn_itr}_itrs/cn.pt')
    true_cost_function = get_true_cost_function(env_id)

    obs_dim = 9
    acs_dim = 3
    feasible_targets = []

    for _ in range(n_goals):
        while True:
            target = np.array([[np.random.uniform(x_start, x_end),
                               np.random.uniform(y_start, y_end),
                               np.random.uniform(z_start, z_end)]])

            obs = np.concatenate(
                [target, np.zeros((target.shape[0], obs_dim - 3))],
                axis=-1)
            action = np.zeros((obs.shape[0], acs_dim))
            cost = true_cost_function(obs, action)
            if cost == 0:
                feasible_targets.append(target)
                break
    feasible_targets = np.array(feasible_targets).squeeze()

    nominal_obs_list, nominal_acs_list, nominal_rew_list, nominal_len_list = [], [], [], []
    for i in range(n_goals):
        target = feasible_targets[i]
        linear_ds = lambda x: A @ (x[:3] - target)
        ds_policy = DS_Policy(3, linear_ds)

        if 'ReachConcave' in env_id:
            ds_policy.modulate_with_NN__with_refer_point(constraint_net.build_gamma_with_grad_for_ds, rho=2)
        elif 'Reach2Region' in env_id:
            ds_policy.modulate_with_NN(constraint_net.build_gamma_with_grad_for_ds, rho=2)
        env = make_eval_env(env_id, use_cost_wrapper=False, normalize_obs=False)
        nominal_obs, _, nominal_acs, nominal_rew, nominal_len = sample_trajectories_for_reach_env(ds_policy, env,
                                                                                                  n_rollouts,
                                                                                                  cost_function=true_cost_function,
                                                                                                  target=target,
                                                                                                  deterministic=True,
                                                                                                  save_dir=save_dir,
                                                                                                  save_namegroup=str(i))
        nominal_obs_list.append(nominal_obs)
        nominal_len_list.append(nominal_len)

    nominal_obs = np.vstack(nominal_obs_list)
    nominal_len = np.concatenate(nominal_len_list)

    plot_constraints(cost_function=constraint_net.cost_function, manual_threshold=0.5, position_limit=None, env=env,
                     env_id=env_id, select_dim=[0, 1, 2], obs_dim=9, acs_dim=3,
                     save_name=save_dir + f'{load_cn_itr}.png', fig_title='Transferred',
                     obs_expert=nominal_obs, rew_expert=None, len_expert=nominal_len, obs_nominal=None,
                     rew_nominal=None, len_nominal=None, policy_excerpt_rollout_num=None, obs_failed=None,
                     obs_rn=None, gmm_learned=None, query_obs=None, show=not False, num_points_for_plot=30)


if __name__=='__main__':

    # generate_policy_with_DS_and_random_goal('ReachConcaveObs-v0', 'run-20240716_093851-0lehfict', '19', n_goals=8, n_rollouts=1)
    generate_policy_with_DS_and_random_goal('Reach2RegionObs-v0', 'run-20240718_041918-4svsme3b', '6', n_goals=8, n_rollouts=1)

    # Generate expert data for pointDS
    # A = np.diag([-1, -1])
    # linear_ds = lambda x: A @ x
    # policy = DS_Policy(2, linear_ds)
    # policy.generate_expert_data_point(save_data=True, failed_demo=False)

    # Generate expert data for point DS with nonlinear DS
    # expert_path = '/home/baiyu/PycharmProjects/icrl-master/icrl/expert_data/PointDS'
    # (x, x_dot, _), length, _ = load_expert_data(expert_path, 99)
    # x_att = np.array([[0., 0.]])
    # x_init = [x[sum(length[:i])].reshape(1, 2) for i in range(len(length))]
    # lpvds = lpvds_class(x, x_dot, x_att)
    # lpvds.logIN(expert_path + '/files/ds.json')
    # nonlinear_ds = lambda x: lpvds.predict(x[np.newaxis, :])
    # policy = DS_Policy(2, nonlinear_ds)
    # policy.generate_demonstrations_PointObs(save_data=False, generate_failed_demo=True)

    # # Generate expert data for ReachConcaveObs
    # A = np.diag([-2, -2, -2])
    # linear_ds = lambda x: A @ (x[:3] - np.array([0.68, 0.0, 0.04]))
    # policy = DS_Policy(3, linear_ds)
    # policy.generate_demonstrations_ReachConcaveObs(failed_demo=False)

    # # Generate expert data for Reach2regionObsObs
    # linear_ds = lambda x: A @ (x[:3] - np.array([0.62, 0.03, 0.04]))
    # policy = DS_Policy(3, linear_ds)
    # policy.generate_demonstrations_Reach2regionsObs()

    # # #-------------------------
    # # Train a gmm constraint model, plot train_data
    # constraint = np.random.randint(0, 2, [20, 1])
    # nominal_obs = np.random.rand(20, 2) * 0.7 + 0.5
    # # nominal_obs = np.random.uniform(-0.1, 0.1, [100,2])
    # idx = np.where(constraint == 1)[0]
    # train_data = nominal_obs[idx]
    # gmm = GaussianMixture(n_components=1, covariance_type='full',
    #                       init_params='kmeans', max_iter=1000,
    #                       tol=0.001, n_init=10)
    # gmm.fit(train_data)
    # gmm.confidence = 5
    # policy.modulate_with_gmm_constraint_2d(gmm)
    #
    # plt.figure(figsize=(10, 10))
    # plt.scatter(train_data[:, 0], train_data[:, 1], s=30, c='red', marker='o', label='Train Data')
    # ax = plt.gca()
    # plot_gmm_ellipse(gmm, ax)
    # plt.scatter(0, 0, color='green', s=30, marker='^', label='Origin')
    # # # # #-------------------------
    # # # # # Plot vector flow
    # # x_start, x_end = -0.5, 1.5
    # # y_start, y_end = -0.5, 1.5
    # #
    # # x = np.arange(x_start, x_end + 0.1, 0.1)
    # # y = np.arange(y_start, y_end + 0.1, 0.1)
    # # x, y = np.meshgrid(x, y)
    # # x_eval = np.vstack([x.ravel(), y.ravel()]).T
    # # fx_eval, _ = policy.predict(x_eval)
    # # plt.quiver(x_eval[:, 0], x_eval[:, 1], fx_eval[:, 0], fx_eval[:, 1], color='blue', scale=40,
    # #            label='Velocity Vectors')
    # # # -------------------------
    # # # Simulate a trajectory
    # # env = make_eval_env('PointDS-v0', False, False)
    # # for _ in range(30):
    # #     env.env_method('set_starting_point', np.random.uniform(0.2, 1.5, [2]))
    # #     observation = env.reset()
    # #     trajectory = []
    # #     for step in range(1000):
    # #         action, _ = policy(observation)  # User-defined policy function
    # #         observation, reward, terminated, info = env.step(action)
    # #         if terminated:
    # #             # print(f'Reached the origin in {step} steps')
    # #             break
    # #         else:
    # #             trajectory.append(observation)
    # #     trajectory = np.vstack(trajectory)
    # #     plt.plot(trajectory[:,0], trajectory[:,1], c='black')
    # plt.legend()
    # plt.show()