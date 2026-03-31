from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
import gym
from matplotlib.ticker import MaxNLocator, MultipleLocator

from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import update_learning_rate
from torch import nn
from tqdm import tqdm
import matplotlib
import pyvista as pv
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler

class ConstraintNet(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            hidden_sizes: Tuple[int, ...],
            batch_size: int,
            lr_schedule: Callable[[float], float],
            is_discrete: bool,
            regularizer_coeff: float = 0.,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            no_importance_sampling: bool = False,
            per_step_importance_sampling: bool = False,
            clip_obs: Optional[float] = 10.,
            initial_obs_mean: Optional[np.ndarray] = None,
            initial_obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            target_kl_old_new: float = -1,
            target_kl_new_old: float = -1,
            train_gail_lambda: Optional[bool] = False,
            eps: float = 1e-5,
            device: str = 'cpu',
            loss_type: str = 'bce',
            sbce_coe: float = 0.3,
            manual_threshold: float = 0.7,
            weight_expert_loss: float = 1.0,
            weight_nominal_loss: float = 1.0,
            initial_feasible_bias: Optional[float] = None,
        ):
        super(ConstraintNet, self).__init__()

        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.obs_select_dim = obs_select_dim
        self.acs_select_dim = acs_select_dim
        self._define_input_dims()
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.is_discrete = is_discrete
        self.regularizer_coeff = regularizer_coeff
        self.importance_sampling = not no_importance_sampling
        self.per_step_importance_sampling = per_step_importance_sampling
        self.clip_obs = clip_obs
        self.device = device
        self.eps = eps
        self.train_gail_lambda = train_gail_lambda
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs['eps'] = self.eps  # a small value to avoid NaN in Adam optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_class = optimizer_class
        self.lr_schedule = lr_schedule
        self.current_obs_mean = initial_obs_mean
        self.current_obs_var = initial_obs_var
        self.action_low = action_low
        self.action_high = action_high
        self.target_kl_old_new = target_kl_old_new
        self.target_kl_new_old = target_kl_new_old
        self.current_progress_remaining = 1.
        self.loss_type = loss_type
        self.sbce_coe = sbce_coe
        self.manual_threshold = manual_threshold
        self.weight_expert_loss = weight_expert_loss
        self.weight_nominal_loss = weight_nominal_loss
        self.initial_feasible_bias = initial_feasible_bias
        # self.weight_decay = 100e-4
        self._build()

    def _define_input_dims(self) -> None:
        self.select_dim = []
        if self.obs_select_dim is None:
            self.select_dim += [i for i in range(self.obs_dim)]
        elif self.obs_select_dim[0] != -1:
            self.select_dim += self.obs_select_dim
        if self.acs_select_dim is None:
            self.select_dim += [i for i in range(self.acs_dim)]
        elif self.acs_select_dim[0] != -1:
            self.select_dim += self.acs_select_dim
        assert len(self.select_dim) > 0, ''
        # self.select_dim = [0,1]
        self.input_dims = len(self.select_dim)
        # print(self.select_dim)

    def _build(self) -> None:
        # creating the network and adding sigmoid at the end
        self.network = nn.Sequential(*create_mlp(self.input_dims, 1, self.hidden_sizes, nn.LeakyReLU), nn.Sigmoid())
        self.network.to(self.device)
        if self.initial_feasible_bias is not None:
            last_linear = None
            for module in self.network.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
            if last_linear is not None:
                nn.init.constant_(last_linear.bias, self.initial_feasible_bias)
        # building the optimizer
        if self.optimizer_class is not None:
            self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr_schedule(1), **self.optimizer_kwargs)
        else:
            self.optimizer = None
        if self.train_gail_lambda:
            self.criterion = nn.BCELoss()

    def forward(self, x: th.tensor) -> th.tensor:
        # if self.table_function: # If using a discrete table to store the value
        #     x = x.clamp(-20, 19.999)
        #     cell_x = (x[:, 0] + 20)//2
        #     cell_y = (x[:, 1] + 20)//2
        #     inx = np.zeros([x.shape[0],400])
        #     hot = (cell_x + cell_y * 20).numpy().astype(int)
        #     for i in range(hot.shape[0]):
        #         inx[i, hot[i]] = 1
        #     return self.network(th.Tensor(inx))
        # else:
        return self.network(x)

    def cost_function(self, obs: np.ndarray, acs: np.ndarray) -> np.ndarray:
        # The cost function for learning the policy, which works like a penalty.
        assert obs.shape[-1] == self.obs_dim, ''
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ''
        x = self.prepare_data(obs, acs)
        with th.no_grad():
            out = self.__call__(x)
        cost = 1 - out.detach().cpu().numpy()
        # line below is the only difference with the orignal cost_function function
        cost = np.where(cost > self.manual_threshold, 1, 0)   # manual thresholding
        return cost.squeeze(axis=-1)

    def cost_function_evaluation(self, obs: np.ndarray, acs: np.ndarray) -> np.ndarray:
        # The cost function for computing the classification accuracy rate. The output value must be 0 or 1.
        assert obs.shape[-1] == self.obs_dim, ''
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ''
        x = self.prepare_data(obs, acs)
        with th.no_grad():
            out = self.__call__(x)
        cost = 1 - out.detach().cpu().numpy()
        cost = np.where(cost > self.manual_threshold, 1, 0)   # The output value must be binary.
        return cost.squeeze(axis=-1)

    def cost_function_non_binary(self, obs: np.ndarray, acs: np.ndarray) -> np.ndarray:
        # Return the original continuous cost function ranging from 0 to 1.
        assert obs.shape[-1] == self.obs_dim, ''
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ''
        x = self.prepare_data(obs, acs)
        with th.no_grad():
            out = self.__call__(x)
        cost = 1 - out.detach().cpu().numpy()
        return cost.squeeze(axis=-1)

    def build_gamma_with_grad_for_ds(self, obs: np.ndarray, acs: np.ndarray) ->  (np.array, np.array):
        # Compute gamma and its gradient, only used for modulating DS
        assert obs.shape[-1] == self.obs_dim, ''
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ''
        x = self.prepare_data(obs, acs)
        x.requires_grad = True
        out = self.__call__(x)
        cost = 1 - out
        grad = torch.autograd.grad(outputs=cost, inputs=x, grad_outputs=torch.ones_like(out))[0]
        gamma = 1 + np.maximum((self.manual_threshold - cost.detach().cpu().numpy()) * 10, 0)
        return gamma, grad.detach().cpu().numpy()

    def call_forward(self, x: np.ndarray):
        with th.no_grad():
            out = self.__call__(th.tensor(x, dtype=th.float32).to(self.device))
        return out

    def train(
            self,
            iterations: np.ndarray,
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            episode_lengths: np.ndarray,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            current_progress_remaining: float = 1,
            is_empty_nominal_obs: bool = False,
        ) -> Dict[str, Any]:
        # This is the original train function from ICRL paper repo, we choose to not change it.
        self._update_learning_rate(current_progress_remaining)

        # updating the normalization stats
        self.current_obs_mean = obs_mean
        self.current_obs_var = obs_var

        # preparing data
        nominal_data = self.prepare_data(nominal_obs, nominal_acs)
        expert_data = self.prepare_data(expert_obs, expert_acs)

        # saving current network predictions if using importance sampling
        if self.importance_sampling:
            with th.no_grad():
                start_preds = self.forward(nominal_data).detach()

        # main loop
        early_stop_itr = iterations
        loss = th.tensor(np.inf)
        for itr in tqdm(range(iterations)):
            # computing the importance sampling (is) weights
            if self.importance_sampling:
                with th.no_grad():
                    current_preds = self.forward(nominal_data).detach()
                is_weights, kl_old_new, kl_new_old = self.compute_is_weights(start_preds.clone(), current_preds.clone(), episode_lengths)
                # breaking if kl is very large
                if ((self.target_kl_old_new != -1 and kl_old_new > self.target_kl_old_new) or
                    (self.target_kl_new_old != -1 and kl_new_old > self.target_kl_new_old)):
                    early_stop_itr = itr
                    break
            else:
                is_weights = th.ones(nominal_data.shape[0])
            # doing a complete pass on data

            for nom_batch_indices, exp_batch_indices in self.get(nominal_data.shape[0], expert_data.shape[0]):
                # getting a batch
                nominal_batch = nominal_data[nom_batch_indices]
                expert_batch = expert_data[exp_batch_indices]
                is_batch = is_weights[nom_batch_indices][..., None].clamp(0, 5)  # hard-coded by Baiyu
                # making predictions
                nominal_preds = self.__call__(nominal_batch)
                expert_preds = self.__call__(expert_batch)
                # calculating the loss
                if self.train_gail_lambda:   # bce
                    nominal_loss = self.criterion(nominal_preds, th.zeros(*nominal_preds.size()))
                    expert_loss = self.criterion(expert_preds, th.ones(*expert_preds.size()))
                    regularizer_loss = th.tensor(0)
                    loss = nominal_loss + expert_loss
                else:   # eq. 7
                    if self.loss_type == 'ml':  # maximum-likelihood loss
                        expert_loss = th.mean(th.log(expert_preds + self.eps))
                        nominal_loss = th.mean(is_batch*th.log(nominal_preds+self.eps))
                        regularizer_loss = (th.mean(expert_preds) + th.mean(nominal_preds))
                    elif self.loss_type == 'bce':   # binary cross-entropy loss
                        expert_loss = th.mean(th.log(expert_preds + self.eps))
                        nominal_loss = -1 * th.mean(th.log(1-nominal_preds+self.eps))

                        # Uniformly sample states from state space and regularize them.
                        # r = np.arange(-10, 10, 0.2)
                        # X, Y = np.meshgrid(r, r)
                        # obs_all = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=-1)
                        # obs_all = np.concatenate((obs_all, np.zeros((np.size(X), expert_obs.shape[1]-2))), axis=-1)
                        # action = np.zeros((np.size(X), expert_acs.shape[1]))
                        # regu_batch = self.prepare_data(obs_all, action)
                        # pred = self.__call__(regu_batch)  # 0 is infeasible
                        # regularizer_loss = pred.mean() * self.regularizer_coeff
                        regularizer_loss = (th.mean(expert_preds) + th.mean(nominal_preds))

                    expert_loss = self.weight_expert_loss * expert_loss
                    nominal_loss = self.weight_nominal_loss * nominal_loss
                    regularizer_loss = self.regularizer_coeff * regularizer_loss
                    loss = -expert_loss + nominal_loss - regularizer_loss

                if not is_empty_nominal_obs:
                    # updating the network
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            # wandb.log({"loss": loss})

        bw_metrics = {'backward/cn_loss': loss.item(),
                      'backward/expert_loss': -expert_loss.item(),
                      'backward/unweighted_nominal_loss': th.mean(th.log(nominal_preds + self.eps)).item(),
                      'backward/nominal_loss': nominal_loss.item(),
                      'backward/regularizer_loss': regularizer_loss.item(),
                      # 'backward/is_mean': th.mean(is_weights).detach().item(),
                      # 'backward/is_max': th.max(is_weights).detach().item(),
                      # 'backward/is_min': th.min(is_weights).detach().item(),
                      # 'backward/nominal_preds_max': th.max(nominal_preds).item(),
                      # 'backward/nominal_preds_min': th.min(nominal_preds).item(),
                      # 'backward/nominal_preds_mean': th.mean(nominal_preds).item(),
                      # 'backward/expert_preds_max': th.max(expert_preds).item(),
                      # 'backward/expert_preds_min': th.min(expert_preds).item(),
                      # 'backward/expert_preds_mean': th.mean(expert_preds).item(),
                      }

        if self.importance_sampling:
            stop_metrics = {'backward/kl_old_new': kl_old_new.item(),
                            'backward/kl_new_old': kl_new_old.item(),
                            'backward/early_stop_itr': early_stop_itr}
            bw_metrics.update(stop_metrics)
        return bw_metrics


    def train_MECL_BC(
            self,
            iterations: np.ndarray,
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            episode_lengths: np.array,
            label_frequency: float
        ) -> Dict[str, Any]:
        # Compared with train function, this function 1. support imbalanced dataset 2. support specifying label frequency (control the weight of expert/nominal in gradient)
        nominal_data = self.prepare_data(nominal_obs, nominal_acs)
        expert_data = self.prepare_data(expert_obs, expert_acs)
        loss = th.tensor(np.inf)
        loss_type = self.loss_type

        if self.importance_sampling:
            with th.no_grad():
                start_preds = self.forward(nominal_data).detach()

        for itr in tqdm(range(iterations)):
            # computing the importance sampling (is) weights
            if self.importance_sampling:
                with th.no_grad():
                    current_preds = self.forward(nominal_data).detach()
                is_weights, kl_old_new, kl_new_old = self.compute_is_weights(start_preds.clone(), current_preds.clone(),
                                                                             episode_lengths)
                # breaking if kl is very large
                if ((self.target_kl_old_new != -1 and kl_old_new > self.target_kl_old_new) or
                        (self.target_kl_new_old != -1 and kl_new_old > self.target_kl_new_old)):
                    early_stop_itr = itr
                    break
            else:
                is_weights = th.ones(nominal_data.shape[0]).to(self.device)

            # doing a complete pass on data. If cn_batch_size is not specified (by default), we use the whole dataset.
            for nom_batch_indices, exp_batch_indices in self.get(nominal_data.shape[0], expert_data.shape[0]):
                # getting a batch
                nominal_batch = nominal_data[nom_batch_indices]
                expert_batch = expert_data[exp_batch_indices]
                is_batch = is_weights[nom_batch_indices][..., None].clamp(0, 5)  # hard-coded by Baiyu
                # making predictions
                nominal_preds = self.__call__(nominal_batch)
                expert_preds = self.__call__(expert_batch)
                # calculating the loss

                if loss_type == 'ml':  # maximum-likelihood loss
                    # n_data_min = min(nominal_preds.shape[0], expert_preds.shape[0])
                    # nominal_preds = nominal_preds[:n_data_min]
                    # expert_preds = expert_preds[:n_data_min]
                    # is_batch =  is_batch[:n_data_min]

                    expert_loss = th.mean(th.log(expert_preds + self.eps))
                    nominal_loss = th.mean(is_batch * th.log(nominal_preds + self.eps))
                    # nominal_loss = -1 * th.mean(th.log(1 - nominal_preds + self.eps))
                    regularizer_loss = self.regularizer_coeff * (th.mean(expert_preds) + th.mean(nominal_preds))

                elif 'bce' in loss_type:  # binary cross-entropy loss
                    self.weight_expert_loss = (2 - label_frequency) / 2
                    self.weight_nominal_loss = label_frequency / 2
                    # By default we first balance the expert and nominal data, but will use imbanlanced dataset if specified
                    if 'imba' in loss_type:
                        n_data = (expert_preds.shape[0] + nominal_preds.shape[0]) / 2
                        expert_loss = th.sum(th.log(expert_preds + self.eps)) / n_data
                        nominal_loss = -1 * th.sum(th.log(1 - nominal_preds + self.eps)) / n_data
                        regularizer_loss = self.regularizer_coeff * (th.sum(expert_preds) + th.sum(nominal_preds)) / n_data

                    else:
                        expert_loss = th.mean(th.log(expert_preds + self.eps))
                        nominal_loss = -1 * th.mean(th.log(1 - nominal_preds + self.eps))
                        regularizer_loss = self.regularizer_coeff * (th.mean(expert_preds) + th.mean(nominal_preds))
                else:
                    raise NotImplementedError('loss type not recognized. Only support ml and bce')

                expert_loss = self.weight_expert_loss * expert_loss
                nominal_loss = self.weight_nominal_loss * nominal_loss
                loss = -expert_loss + nominal_loss - regularizer_loss

                # updating the network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if nominal_obs.size == 0:
            bw_metrics = {}
        else:
            bw_metrics = {'backward/cn_loss': loss.item(),
                          'backward/nominal_preds_max': th.max(nominal_preds).item(),
                          'backward/nominal_preds_min': th.min(nominal_preds).item(),
                          'backward/nominal_preds_mean': th.mean(nominal_preds).item(),
                          'backward/expert_preds_max': th.max(expert_preds).item(),
                          'backward/expert_preds_min': th.min(expert_preds).item(),
                          'backward/expert_preds_mean': th.mean(expert_preds).item(),
                          }
        return bw_metrics

    def train_with_MIL(
            self,
            iterations: np.ndarray,
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            expert_len: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            nominal_len: np.ndarray,
            mil_config: object,
    ) -> Dict[str, Any]:
        expert_data = self.prepare_data(expert_obs, expert_acs)
        nominal_data = self.prepare_data(nominal_obs, nominal_acs)
        device = expert_data.device

        nominal_traj_labels = torch.zeros(len(nominal_len), 1, device=device)
        expert_traj_labels = torch.ones(len(expert_len), 1, device=device)
        combined_traj_labels = torch.cat((nominal_traj_labels, expert_traj_labels), dim=0)
        combined_data = torch.cat((nominal_data, expert_data), dim=0)
        self.network.train()

        def pool_traj_preds(preds_traj: torch.Tensor, temperature: float) -> torch.Tensor:
            preds_traj = preds_traj.clamp(self.eps, 1 - self.eps)
            if mil_config.cn_mil_pooling == 'lse':
                scaled_infeasibility = (1 - preds_traj) / temperature
                log_mean_exp = torch.logsumexp(scaled_infeasibility, dim=0) - torch.log(
                    torch.tensor(preds_traj.shape[0], device=preds_traj.device, dtype=preds_traj.dtype)
                )
                return 1 - temperature * log_mean_exp
            if mil_config.cn_mil_pooling == 'mean':
                return preds_traj.mean()
            if mil_config.cn_mil_pooling == 'min':
                return preds_traj.min()
            if mil_config.cn_mil_pooling == 'noiseor':
                return preds_traj.prod()
            raise NotImplementedError(
                'Pooling method ' + str(mil_config.cn_mil_pooling) + ' not recognized'
            )

        loss = torch.tensor(float('inf'), device=device)
        nominal_preds = torch.empty(0, device=device)
        expert_preds = torch.empty(0, device=device)

        for itr in tqdm(range(iterations)):
            preds = self.__call__(combined_data).squeeze(-1)

            combined_traj_preds = []
            start_idx = 0
            temperature = mil_config.cn_mil_Tn

            for length in nominal_len:
                end_idx = start_idx + int(length)
                preds_traj = preds[start_idx:end_idx]
                combined_traj_preds.append(pool_traj_preds(preds_traj, temperature))
                start_idx = end_idx

            nominal_end_idx = start_idx
            for length in expert_len:
                end_idx = start_idx + int(length)
                preds_traj = preds[start_idx:end_idx]
                combined_traj_preds.append(pool_traj_preds(preds_traj, temperature))
                start_idx = end_idx

            combined_traj_preds = torch.stack(combined_traj_preds).unsqueeze(1)
            combined_traj_preds = combined_traj_preds.clamp(self.eps, 1 - self.eps)
            loss = nn.BCELoss()(combined_traj_preds, combined_traj_labels)
            if ((itr + 1) % 100 == 0) or (itr == iterations - 1):
                print(f'MIL backward iteration {itr + 1}/{iterations}, loss: {loss.item():.6f}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            nominal_preds = preds[:nominal_end_idx]
            expert_preds = preds[nominal_end_idx:]

        self.network.eval()

        if nominal_obs.size == 0:
            return None, nominal_obs, {'backward/cn_loss': loss.item()}

        bw_metrics = {
            'backward/cn_loss': loss.item(),
            'backward/nominal_preds_max': th.max(nominal_preds).item(),
            'backward/nominal_preds_min': th.min(nominal_preds).item(),
            'backward/nominal_preds_mean': th.mean(nominal_preds).item(),
            'backward/expert_preds_max': th.max(expert_preds).item(),
            'backward/expert_preds_min': th.min(expert_preds).item(),
            'backward/expert_preds_mean': th.mean(expert_preds).item(),
        }
        return None, nominal_obs, bw_metrics

    def train_with_two_step_pu_learning(
            self,
            iterations: np.ndarray,
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            nominal_len: np.array,
            pu_config: object,
    ) -> Dict[str, Any]:

        # First -step ,selecting appropriate reliable infeasible data
        nominal_data = self.select_appropriate_dims(np.concatenate([nominal_obs, nominal_acs], axis=-1))
        expert_data = self.select_appropriate_dims(np.concatenate([expert_obs, expert_acs], axis=-1))
        # Identify the reliable nominal data only if the nominal data is not empty
        if nominal_data.size != 0:
            if pu_config.rn_decision_method == 'GPU':   # Generative PU learning

                print("Running GPU method for reliable negative selection")
                # Select best gmm model by BIC
                n_components = np.arange(3, pu_config.GPU_n_gmm)
                models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(expert_data) for n in n_components]
                BIC_scores = [m.bic(expert_data) for m in models]
                model_index = np.argmin(BIC_scores)

                # plt.figure(figsize=(10, 6))
                # plt.plot(np.arange(3, pu_config.GPU_n_gmm), BIC_scores, marker='o')
                # plt.title("BIC Scores vs Number of Components")
                # plt.xlabel("Number of Components")
                # plt.ylabel("BIC Score")
                # plt.grid(True)
                # plt.savefig('icrl/tests/pu_learning/BIC.png')

                expert_distribution = models[model_index]
                expert_distribution.confidence = 5
                print(f"Selected number of components: {expert_distribution.n_components}")
                nominal_likelihood = expert_distribution.score_samples(nominal_data)
                reliable_negative_threshold = pu_config.GPU_likelihood_thresh
                rn_nominal_index = nominal_likelihood < reliable_negative_threshold

                # Add at least on RN point from each trajectory
                if pu_config.add_rn_each_traj and rn_nominal_index.any() == True:
                    knn = NearestNeighbors(n_neighbors=pu_config.kNN_k)
                    knn.fit(nominal_data[rn_nominal_index])
                    distances, indices = knn.kneighbors(nominal_data)
                    distances_sum = np.mean(distances, axis=1)

                    start_idx = 0
                    for traj_len in nominal_len:
                        end_idx = start_idx + int(traj_len)
                        traj_distances = distances_sum[start_idx:end_idx]
                        min_dist_idx = np.argmin(traj_distances)
                        rn_nominal_index[start_idx + min_dist_idx] = True

                        first_false_index = np.argmin(rn_nominal_index[start_idx:end_idx])
                        rn_nominal_index[start_idx:start_idx + first_false_index] = False

                        start_idx = end_idx
            elif pu_config.rn_decision_method == 'CPU': # Clustering-based PU method

                # Merge the datasets
                merged_data = np.vstack((nominal_data, expert_data))
                nominal_indices = np.arange(nominal_data.shape[0])
                expert_indices = np.arange(nominal_data.shape[0], merged_data.shape[0])

                # Use gmm for clustering, decide the best number of components by BIC
                n_components = np.arange(6, pu_config.CPU_n_gmm_k)
                models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(merged_data) for n in
                          n_components]
                BIC_scores = [m.bic(merged_data) for m in models]
                model_index = np.argmin(BIC_scores)
                gmm = models[model_index]
                print(f"Selected number of components: {gmm.n_components}")
                merged_data = np.vstack((nominal_data, expert_data))
                clusters = gmm.predict(merged_data)

                cluster_proportions = []
                for cluster_id in np.unique(clusters):
                    cluster_indices = np.where(clusters == cluster_id)[0]
                    nominal_count = np.intersect1d(cluster_indices, nominal_indices).size
                    total_count = cluster_indices.size
                    proportion_nominal = nominal_count / total_count
                    cluster_proportions.append(proportion_nominal)
                high_nominal_clusters = [cluster_id for cluster_id, prop in enumerate(cluster_proportions) if
                                         prop > pu_config.CPU_ratio_thresh]
                rn_nominal_index = np.array([clusters[i] in high_nominal_clusters for i in nominal_indices])
                expert_distribution = gmm
                gmm.confidence = 5

            elif pu_config.rn_decision_method == 'kNN': # kNN PU learning

                # Normalize the data if needed
                if pu_config.kNN_normalize:
                    combined_data = np.vstack((expert_data, nominal_data))
                    scaler = StandardScaler()
                    normalized_combined_data = scaler.fit_transform(combined_data)
                    n_expert = expert_data.shape[0]
                    expert_data = normalized_combined_data[:n_expert, :]
                    nominal_data = normalized_combined_data[n_expert:, :]

                    # Show normalized data distribution
                    # n_dims = expert_data.shape[1]
                    # fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims))
                    # fig.tight_layout(pad=4.0)
                    #
                    # if n_dims == 1:
                    #     axes = [axes]
                    # for i in range(n_dims):
                    #     axes[i].hist(expert_data[:, i], bins=30, color='blue', alpha=0.5, label='Expert Data',
                    #                  edgecolor='black', density=True)
                    #     axes[i].hist(nominal_data[:, i], bins=30, color='orange', alpha=0.5, label='Nominal Data',
                    #                  edgecolor='black', density=True)
                    #     axes[i].set_title(f'Dimension {i + 1} Distribution')
                    #     axes[i].set_xlabel(f'Dimension {i + 1}')
                    #     axes[i].set_ylabel('Density')
                    #     axes[i].legend()
                    # plt.show()

                # Create kNN model with specified metric
                if pu_config.kNN_metric == 'euclidean':
                    knn = NearestNeighbors(n_neighbors=pu_config.kNN_k, p=2)
                elif pu_config.kNN_metric == 'manhattan':
                    knn = NearestNeighbors(n_neighbors=pu_config.kNN_k, p=1)
                elif pu_config.kNN_metric == 'chebyshev':
                    knn = NearestNeighbors(n_neighbors=pu_config.kNN_k, p=np.inf)
                elif pu_config.kNN_metric == 'weighted_euclidean':
                    # Need to control weight manually. By default, we give large weights to the first two dimensions
                    num_dims = nominal_data.shape[1]
                    w = np.zeros(num_dims)
                    w[0] = w[1] = 0.4
                    remaining_weight = 0.2 / (num_dims - 2)
                    w[2:] = remaining_weight
                    expert_data = expert_data * w**0.5
                    nominal_data = nominal_data * w**0.5
                    knn = NearestNeighbors(n_neighbors=pu_config.kNN_k, p=2)

                knn.fit(expert_data)
                distances, indices = knn.kneighbors(nominal_data)
                distances_sum = np.mean(distances, axis=1)
                rn_nominal_index = distances_sum > pu_config.kNN_thresh

                # Add the most reliable infeasible data from each trajectory
                if pu_config.add_rn_each_traj and rn_nominal_index.any()==True:
                    knn.fit(nominal_data[rn_nominal_index])
                    distances, indices = knn.kneighbors(nominal_data)
                    distances_sum = np.mean(distances, axis=1)

                    start_idx = 0
                    for traj_len in nominal_len:
                        end_idx = start_idx + int(traj_len)
                        traj_distances = distances_sum[start_idx:end_idx]
                        min_dist_idx = np.argmin(traj_distances)
                        rn_nominal_index[start_idx + min_dist_idx] = True

                        first_false_index = np.argmin(rn_nominal_index[start_idx:end_idx])
                        rn_nominal_index[start_idx:start_idx+first_false_index] = False

                        start_idx = end_idx
                expert_distribution = None
            # elif pu_config.rn_decision_method == 'WB':
            #     rn_nominal_index = np.full(nominal_obs.shape[0], True)
            #     label_frequency = pu_config.WB_label_frequency
            #     self.weight_expert_loss = (2 - label_frequency)/2
            #     self.weight_nominal_loss = label_frequency/2
            #     expert_distribution = None
        else:
            # If no nominal data, no rn nominal
            rn_nominal_index = np.zeros(0, dtype=bool)
            expert_distribution = None
        # Creat datasets in pytorch
        nominal_data = self.prepare_data(nominal_obs, nominal_acs)
        expert_data = self.prepare_data(expert_obs, expert_acs)

        #  The second step -------------------
        # Learning the constraint model.
        # If refine_iterations is greater than 1, the model will expand the RN data and retrain the model.
        for _ in range(pu_config.refine_iterations):
            rn_nominal_data = nominal_data[rn_nominal_index]
            device = expert_data.device
            nominal_labels = torch.zeros(rn_nominal_data.shape[0], 1, device=device)
            expert_labels = torch.ones(expert_data.shape[0], 1, device=device)
            combined_data = torch.cat((rn_nominal_data, expert_data), dim=0)
            combined_labels = torch.cat((nominal_labels, expert_labels), dim=0)
            for _ in tqdm(range(iterations)):
                for batch_indices in self.get_combined(combined_data.shape[0]):
                    # getting a batch
                    batch = combined_data[batch_indices]
                    label = combined_labels[batch_indices]
                    # making predictions
                    preds = self.__call__(batch)
                    loss_function = nn.BCELoss()
                    loss = loss_function(preds, label)

                    # updating the network
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            with torch.no_grad():
                nominal_preds = self.__call__(nominal_data)
                expert_preds = self.__call__(expert_data)
            # rn_nominal_index = np.logical_or(rn_nominal_index, (nominal_preds<0.5).squeeze().cpu().numpy())

        if nominal_obs.size == 0:
            bw_metrics = {}
        else:
            bw_metrics = {'backward/cn_loss': loss.item(),
                          'backward/nominal_preds_max': th.max(nominal_preds).item(),
                          'backward/nominal_preds_min': th.min(nominal_preds).item(),
                          'backward/nominal_preds_mean': th.mean(nominal_preds).item(),
                          'backward/expert_preds_max': th.max(expert_preds).item(),
                          'backward/expert_preds_min': th.min(expert_preds).item(),
                          'backward/expert_preds_mean': th.mean(expert_preds).item(),
                          }

        rn_nominal_obs = nominal_obs[rn_nominal_index]
        return expert_distribution, rn_nominal_obs, bw_metrics # Return the expert distribution fitted by GMM for future plotting

    def train_with_binary_classification(self, expert_obs, expert_acs, nominal_obs, nominal_acs, iterations):
        # Train constraint model with standard binary classification. Not used in PUCL project

        # Creat datasets in pytorch
        nominal_data = self.prepare_data(nominal_obs, nominal_acs)
        expert_data = self.prepare_data(expert_obs, expert_acs)
        for _ in tqdm(range(iterations)):
            device = expert_data.device
            nominal_labels = torch.zeros(nominal_data.shape[0], 1, device=device)
            expert_labels = torch.ones(expert_data.shape[0], 1, device=device)
            combined_data = torch.cat((nominal_data, expert_data), dim=0)
            combined_labels = torch.cat((nominal_labels, expert_labels), dim=0)
            for batch_indices in self.get_combined(combined_data.shape[0]):
                # getting a batch
                batch = combined_data[batch_indices]
                label = combined_labels[batch_indices]
                # making predictions
                preds = self.__call__(batch)
                loss_function = nn.BCELoss()
                loss = loss_function(preds, label)

                # updating the network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def compute_is_weights(self, preds_old: th.Tensor, preds_new: th.Tensor, episode_lengths: np.ndarray) -> th.tensor:
        # computing the importance sampling weights
        with th.no_grad():
            n_episodes = len(episode_lengths)
            cumulative = [0] + list(accumulate(episode_lengths))
            ratio = (preds_new+self.eps) / (preds_old+self.eps) # eq. 10
            prod = [th.prod(ratio[cumulative[j]:cumulative[j+1]]) for j in range(n_episodes)]
            prod = th.tensor(prod)
            normed = n_episodes*prod/(th.sum(prod)+self.eps)
            if self.per_step_importance_sampling:
                is_weights = th.tensor(ratio/th.mean(ratio))
            else:
                is_weights = []
                for length, weight in zip(episode_lengths, normed):
                    is_weights += [weight] * length
                is_weights = th.tensor(is_weights)
            # computing kl(old, current)
            kl_old_new = th.mean(-th.log(prod+self.eps))
            # computing kl(current, old)
            prod_mean = th.mean(prod)
            kl_new_old = th.mean((prod-prod_mean)*th.log(prod+self.eps)/(prod_mean+self.eps))
        return is_weights.to(self.device), kl_old_new, kl_new_old

    def prepare_data(self, obs: np.ndarray, acs: np.ndarray) -> th.tensor:
        obs = self.normalize_obs(obs, self.current_obs_mean, self.current_obs_var, self.clip_obs)
        acs = self.reshape_actions(acs)
        acs = self.clip_actions(acs, self.action_low, self.action_high)
        concat = self.select_appropriate_dims(np.concatenate([obs, acs], axis=-1))
        return th.tensor(concat, dtype=th.float32).to(self.device)

    def select_appropriate_dims(self, x: Union[np.ndarray, th.tensor]) -> Union[np.ndarray, th.tensor]:
        return x[...,self.select_dim]

    def normalize_obs(self, obs: np.ndarray, mean: Optional[float] = None, var: Optional[float] = None,
                      clip_obs: Optional[float] = None) -> np.ndarray:
        if mean is not None and var is not None:
            mean, var = mean[None], var[None]
            obs = (obs-mean) / np.sqrt(var+self.eps)
        if clip_obs is not None:
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        return obs

    def reshape_actions(self, acs):
        if self.is_discrete:
            acs_ = acs.astype(int)
            if len(acs.shape) > 1:
                acs_ = np.squeeze(acs_, axis=-1)
            acs = np.zeros([acs.shape[0], self.acs_dim])
            acs[np.arange(acs_.shape[0]), acs_] = 1.
        return acs

    def clip_actions(self, acs: np.ndarray, low: Optional[float] = None, high: Optional[float] = None) -> np.ndarray:
        if high is not None and low is not None:
            acs = np.clip(acs, low, high)
        return acs

    def get(self, nom_size: int, exp_size: int) -> np.ndarray:
        if self.batch_size is None:
            yield np.arange(nom_size), np.arange(exp_size)
        else:
            # I changed here to generate indices different for each class
            size = min(nom_size, exp_size)
            # indices = np.random.permutation(size)
            nom_indices = np.random.permutation(nom_size)
            exp_indices = np.random.permutation(exp_size)
            batch_size = self.batch_size
            # returnnig everything, do not create minibatches
            if batch_size is None:
                batch_size = size
            start_idx = 0
            while start_idx < size:
                nom_batch_indices = nom_indices[start_idx:start_idx+batch_size]
                exp_batch_indices = exp_indices[start_idx:start_idx+batch_size]
                yield nom_batch_indices, exp_batch_indices
                start_idx += batch_size

    def get_combined(self, size: int) -> np.ndarray:
        if self.batch_size is None:
            yield np.arange(size)
        else:
            indices = np.random.permutation(size)
            batch_size = self.batch_size
            start_idx = 0
            while start_idx < size:
                batch_indices = indices[start_idx:start_idx+batch_size]
                yield batch_indices
                start_idx += batch_size


    def _update_learning_rate(self, current_progress_remaining) -> None:
        self.current_progress_remaining = current_progress_remaining
        update_learning_rate(self.optimizer, self.lr_schedule(current_progress_remaining))

    def save(self, save_path):
        state_dict = dict(
                cn_network=self.network.state_dict(),
                cn_optimizer=self.optimizer.state_dict(),
                obs_dim=self.obs_dim,
                acs_dim=self.acs_dim,
                is_discrete=self.is_discrete,
                obs_select_dim=self.obs_select_dim,
                acs_select_dim=self.acs_select_dim,
                clip_obs=self.clip_obs,
                obs_mean=self.current_obs_mean,
                obs_var=self.current_obs_var,
                action_low=self.action_low,
                action_high=self.action_high,
                device=self.device,
                hidden_sizes=self.hidden_sizes,
                initial_feasible_bias=self.initial_feasible_bias,
        )
        th.save(state_dict, save_path)

    def _load(self, load_path):
        state_dict = th.load(load_path)
        if 'cn_network' in state_dict:
            self.network.load_state_dict(state_dict['cn_network'])
        if 'cn_optimizer' in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict['cn_optimizer'])

    # providing basic functionality to load this class
    @classmethod
    def load(
            cls,
            load_path: str,
            obs_dim: Optional[int] = None,
            acs_dim: Optional[int] = None,
            is_discrete: bool = None,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            clip_obs: Optional[float] = None,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            device: str = 'cpu',
        ):

        state_dict = th.load(load_path)
        # if value inputs are not specified, then get them from the state_dict
        if obs_dim is None:
            obs_dim = state_dict['obs_dim']
        if acs_dim is None:
            acs_dim = state_dict['acs_dim']
        if is_discrete is None:
            is_discrete = state_dict['is_discrete']
        if obs_select_dim is None:
            obs_select_dim = state_dict['obs_select_dim']
        if acs_select_dim is None:
            acs_select_dim = state_dict['acs_select_dim']
        if clip_obs is None:
            clip_obs = state_dict['clip_obs']
        if obs_mean is None:
            obs_mean = state_dict['obs_mean']
        if obs_var is None:
            obs_var = state_dict['obs_var']
        if action_low is None:
            action_low = state_dict['action_low']
        if action_high is None:
            action_high = state_dict['action_high']
        initial_feasible_bias = state_dict.get('initial_feasible_bias')
        if device is None:
            device = state_dict['device']
        # creating the network
        hidden_sizes = state_dict['hidden_sizes']
        constraint_net = cls(
                obs_dim, acs_dim, hidden_sizes, None, None, optimizer_class=None,
                is_discrete=is_discrete, obs_select_dim=obs_select_dim, acs_select_dim=acs_select_dim,
                clip_obs=clip_obs, initial_obs_mean=obs_mean, initial_obs_var=obs_var, action_low=action_low, action_high=action_high,
                device=device, initial_feasible_bias=initial_feasible_bias
        )
        constraint_net.network.load_state_dict(state_dict['cn_network'])
        return constraint_net


# =====================================================================
# plotting utilities
# =====================================================================


def plot_cylinder(ax, center, radius, height, color, alpha):
    z = np.linspace(0, 0 + height, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = center[0] + radius * np.cos(theta_grid)
    y_grid = center[1] + radius * np.sin(theta_grid)

    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, rstride=5, cstride=5)

def plot_gmm_ellipse(gmm, ax, **kwargs):
    for means, covariances in zip(gmm.means_, gmm.covariances_):
        vals, vecs = np.linalg.eigh(covariances)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals * gmm.confidence)
        ellipse = Ellipse(xy=means, width=width, height=height, angle=theta, fc='None', lw=1.5, **kwargs)
        ellipse_handle = ax.add_patch(ellipse)
    return ellipse_handle

def plot_constraints(cost_function, manual_threshold, position_limit, env, env_id, select_dim, obs_dim, acs_dim,
                     save_name, fig_title, obs_expert, rew_expert, len_expert,
                     obs_nominal=None, rew_nominal=None, len_nominal=None, policy_excerpt_rollout_num=None, obs_failed=None, obs_rn=None,
                      gmm_learned=None, query_obs=None, show=False, **kwargs
                     ):
    if show:
        matplotlib.use('TKAgg')
    else:
        matplotlib.use('Agg')

    if env_id in ['PointEnv-v0', 'PointEnvTest-v0', 'PointNullRewardTest-v0', 'PointCircleTest-v0', 'PointObstacle2Test-v0', 'PointHalfCircleTest-v0', 'PointObstacleTest-v0'
                  , 'SwimmerWithPosTest-v0', 'AntWall-v0', 'AntWallTest-v0']:
        #TODO: optimize the code
        if (policy_excerpt_rollout_num == None) or (policy_excerpt_rollout_num) == 0:
            policy_excerpt_rollout_num = len(len_expert)
        plot_for_gym_envs(cost_function, manual_threshold, position_limit, env, env_id, select_dim, obs_dim, acs_dim,
                          save_name, fig_title, obs_expert, rew_expert, len_expert, obs_nominal, rew_nominal, len_nominal, policy_excerpt_rollout_num, obs_failed)
    elif env_id in ['HCWithPosTest-v0', 'HCWithPos-v0']:
        plot_for_halfcheetah(cost_function, manual_threshold, obs_dim, acs_dim, select_dim,
                             save_name, fig_title, obs_nominal, obs_expert, len_expert, obs_rn)
    elif 'UR5' in env_id:
        if (policy_excerpt_rollout_num == None) or (policy_excerpt_rollout_num) == 0:
            policy_excerpt_rollout_num = len(len_expert)
        # plot_for_UR5_envs(cost_function, manual_threshold, position_limit, env, env_id, select_dim, obs_dim, acs_dim,
        #                   save_name, fig_title, obs_expert, rew_expert, len_expert, obs_nominal, rew_nominal, len_nominal, policy_excerpt_rollout_num, obs_failed)
        plot_for_UR5_envs_3D(cost_function, manual_threshold, acs_dim, save_name, fig_title, obs_nominal, len_nominal)
    elif 'PointDS' in env_id or 'PointEllip' in env_id:
        point_env = gym.make(env_id)
        gmm_true = point_env.get_gmm
        plot_for_PointObs(cost_function, manual_threshold, obs_dim, acs_dim, save_name, fig_title, obs_nominal, obs_expert, obs_rn, gmm_true, gmm_learned, query_obs, show=show)
    elif 'ReachVelObs' in env_id or 'ReachVel' in env_id:
        plot_for_ReachVel(cost_function, manual_threshold, obs_dim, acs_dim, select_dim, save_name, fig_title,
                          obs_nominal, obs_expert, len_expert, obs_rn)
    elif 'ReachObs' in env_id or 'Reach2Region' in env_id or 'ReachConcave' in env_id:
        plot_for_ReachObs(cost_function, manual_threshold, obs_dim, acs_dim, select_dim, save_name, fig_title,
                          obs_nominal, obs_expert, len_expert, len_nominal, obs_rn, **kwargs)
    else:
        raise NotImplementedError(f"Plot function for {env_id} is not implemented yet.")

    if show:
        plt.show()
    plt.close('all')

def plot_for_gym_envs(cost_function, manual_threshold, position_limit, env, env_id, select_dim, obs_dim, acs_dim,
                      save_name, fig_title, obs_e, rew_e, len_e, obs_n, rew_n, len_n, n_policy, obs_f=None):
    # TODO:to be optimized
    if len(select_dim) > 2:
        fig, ax = plt.subplots(1, 1, figsize=(15, position_limit))

        obs_e = np.clip(obs_e, -position_limit, position_limit)
        start = 0
        for i, length in enumerate(len_e):
            expert_tra, = ax.plot(obs_e[np.arange(start, start + length - 1, 2), 0],
                                  obs_e[np.arange(start, start + length - 1, 2), 1], clip_on=False,
                                  color='limegreen', linewidth=3, ls='--')  # color=[0, 0.3+rewards_weight[i]*0.7, 0.]
            ax.plot(obs_e[start, 0], obs_e[start, 1], clip_on=False, marker='.', markersize=20, color='green')
            ax.plot(obs_e[start + length - 2, 0], obs_e[start + length - 2, 1], clip_on=False, marker='.',
                    markersize=20, color='dodgerblue')
            start += length
        if obs_n is not None and obs_n != []:
            obs_n = np.clip(obs_n, -position_limit, position_limit)
            nominal_tra = None
            if len_n is None:
                ax.scatter(obs_n[..., 0], obs_n[..., 1], clip_on=False, c='g')
            else:
                start = 0
                memory_tra = None
                for i_traj, length in enumerate(len_n):
                    if length == 0:
                        continue
                    if i_traj < n_policy:
                        nominal_tra, = ax.plot(obs_n[np.arange(start, start + length - 1, 1), 0],
                                               obs_n[np.arange(start, start + length - 1, 1), 1], clip_on=False,
                                               color='orangered',  # [0, min(0.3 + rewards_weight[i] * 0.7, 1), 0.],   #
                                               linewidth=3)
                        ax.plot(obs_n[start, 0], obs_n[start, 1], clip_on=False, marker='.', markersize=20,
                                color='white')
                        ax.plot(obs_n[start + length - 2, 0], obs_n[start + length - 2, 1], clip_on=False, marker='.',
                                markersize=20, color='black')
                    else:
                        memory_tra = ax.scatter(obs_n[np.arange(start, start + length - 1, 1), 0],
                                                obs_n[np.arange(start, start + length - 1, 1), 1],
                                                # clip_on=False,
                                                color='yellow',  # [0, min(0.3 + rewards_weight[i] * 0.7, 1), 0.],   #
                                                # linewidth=3, ls='--'
                                                )
                    start += length
            if nominal_tra is not None:
                ax.legend([expert_tra, nominal_tra, memory_tra], ['Demonstration', 'Policy', 'Memory'], fontsize=20,
                          loc='lower right')
        else:
            ax.legend([expert_tra], ['Demonstration'], fontsize=20, loc='lower right')
        ax.set_ylim([-position_limit, position_limit])
        ax.set_xlim([-position_limit, position_limit])
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_ylim([-13, 13])
        ax.set_xlim([-13, 13])
        ax.set_xlabel('$x$', fontsize=20)
        ax.set_ylabel('$y$', fontsize=20)

        ax.tick_params(labelsize=20)
        plt.grid('on')
        plt.title(fig_title, fontsize=30)
        fig.savefig(save_name, bbox_inches='tight')
        plt.close(fig=fig)

    # condition for the len(select_dim) == 1 has been removed by Erfan.
    if len(select_dim) == 2:
        fig, ax = plt.subplots(1, 1, figsize=(15, position_limit))
        r = np.arange(-position_limit, position_limit, 0.1) # hard-coded by Baiyu
        X, Y = np.meshgrid(r, r)
        obs_all = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=-1)
        obs_all = np.concatenate((obs_all, np.zeros((np.size(X), obs_dim-2))), axis=-1)
        action = np.zeros((np.size(X), acs_dim))
        outs = 1 - cost_function(obs_all, action)
        im = ax.imshow(outs.reshape(X.shape), extent=[-position_limit, position_limit, -position_limit, position_limit],
                       cmap='jet_r', vmin=0, vmax=1, origin='lower')
        contour = plt.contour(X, Y, outs.reshape(X.shape), [1-manual_threshold], colors='k', linewidths=2.0)
        contour1 = ax.contour(X, Y, outs.reshape(X.shape), [0.5], colors='k', linewidths=2.0, linestyles='--')
        plt.clabel(contour1, fontsize=15, colors=('k'))
        plt.clabel(contour, fontsize=15, colors=('k'))
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(labelsize=20)

        obs_e = np.clip(obs_e, -position_limit, position_limit)
        start = 0
        for i, length in enumerate(len_e):
            expert_tra, = ax.plot(obs_e[np.arange(start, start+length-1, 2), 0],
                                  obs_e[np.arange(start, start+length-1, 2), 1], clip_on=False,
                                  color='limegreen', linewidth=3, ls='--')   # color=[0, 0.3+rewards_weight[i]*0.7, 0.]
            ax.plot(obs_e[start, 0], obs_e[start, 1], clip_on=False, marker='.', markersize=20, color='green')
            ax.plot(obs_e[start+length-2, 0], obs_e[start+length-2, 1], clip_on=False, marker='.', markersize=20, color='dodgerblue')
            start += length
        if obs_n is not None and obs_n != [] and obs_n.size != 0:
            obs_n = np.clip(obs_n, -position_limit, position_limit)
            nominal_tra = None

            if len_n is None:
                ax.scatter(obs_n[...,0], obs_n[...,1], clip_on=False, c='g')
            else:
                start = 0
                memory_tra = None
                for i_traj, length in enumerate(len_n):
                    if length == 0:
                        continue
                    if i_traj < n_policy:
                        if not isinstance(length.item(), int):
                            print(f"Value  is not an integer. Exiting loop.")
                        nominal_tra, = ax.plot(obs_n[np.arange(start, start + length-1, 1), 0],
                                   obs_n[np.arange(start, start + length-1, 1), 1], clip_on=False,
                                   color='orangered',#[0, min(0.3 + rewards_weight[i] * 0.7, 1), 0.],   #
                                               linewidth=3)
                        ax.plot(obs_n[start, 0], obs_n[start, 1], clip_on=False, marker='.', markersize=20,
                                color='white')
                        ax.plot(obs_n[start + length - 2, 0], obs_n[start + length - 2, 1], clip_on=False, marker='.',
                                markersize=20, color='black')
                    else:
                        memory_tra = ax.scatter(obs_n[np.arange(start, start + length - 1, 1), 0],
                                               obs_n[np.arange(start, start + length - 1, 1), 1],
                                                 # clip_on=False,
                                                color='yellow',  # [0, min(0.3 + rewards_weight[i] * 0.7, 1), 0.],   #
                                               # linewidth=3, ls='--'
                                                 )
                        # memory_tra, = ax.plot(obs_n[np.arange(start, start + length - 1, 1), 0],
                        #                       obs_n[np.arange(start, start + length - 1, 1), 1], clip_on=False,
                        #                       color='orangered',  # [0, min(0.3 + rewards_weight[i] * 0.7, 1), 0.],   #
                        #                       linewidth=3, ls='--')

                    start += length
            if nominal_tra is not None:
                ax.legend([expert_tra, nominal_tra, memory_tra], ['Demonstration', 'Policy', 'Memory'], fontsize=20, loc='lower right')
        else:
            ax.legend([expert_tra], ['Demonstration'], fontsize=20, loc='lower right')

        ax.set_ylim([-position_limit, position_limit])
        ax.set_xlim([-position_limit, position_limit])

        if 'PointObstacle' in env_id:
            ax.set_xlim([-1, 8])
            ax.set_ylim([-7.5, 10.5])

        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)
        ax.set_xlabel('$x$', fontsize=20)
        ax.set_ylabel('$y$', fontsize=20)

        ax.tick_params(labelsize=20)
        plt.grid('on')
        plt.title(fig_title, fontsize=30)
        fig.savefig(save_name, bbox_inches='tight')
        plt.close(fig=fig)

def plot_for_halfcheetah(cost_function, manual_threshold, obs_dim, acs_dim, cn_obs_select_dim, save_name, fig_title, obs_n, obs_e, len_e, obs_rn):

    x_lim = [-13, 13]
    y_lim = [-1, 1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    if  (cost_function is not None) and len(cn_obs_select_dim) == 2:
        r1 = np.arange(x_lim[0], x_lim[1], 0.1)
        r2 = np.arange(y_lim[0], y_lim[1], 0.1)
        X, Y = np.meshgrid(r1, r2)

        obs = np.concatenate(
            [X.reshape([-1, 1]), Y.reshape([-1, 1]), np.zeros((X.size, obs_dim - 2))], axis=-1)
        action = np.zeros((np.size(X), acs_dim))
        outs = (1 - cost_function(obs, action)).reshape(X.shape)

        im = ax.imshow(outs.reshape(X.shape), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]],
                       cmap='jet_r', vmin=0, vmax=1, origin='lower')
        contour = ax.contour(X, Y, outs.reshape(X.shape), [1 - manual_threshold], colors='k', linewidths=2.0)
        plt.clabel(contour, fontsize=8, colors=('k'))

        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(labelsize=10)

    #
    # else:
        # if obs_n is not None and obs_n.size != 0:
        #     action = np.zeros((obs_n.shape[0], acs_dim))
        #     cost = (1 - cost_function(obs_n, action)).reshape(-1, 1)
        #     feasibility = (cost < manual_threshold)
        #     ax.scatter(obs_n[..., 0], obs_n[..., 1], clip_on=False, c=feasibility, cmap='jet_r', edgecolors='k', s=1)
    if obs_n is not None and obs_n.size != 0:
        ax.scatter(obs_n[..., 0], obs_n[..., 1], clip_on=False, c='orange', edgecolors='orange', label='policy',
                   s=1)
    # if obs_rn is not None and obs_rn.size != 0:
    #     ax.scatter(obs_rn[..., 0], obs_rn[..., 1], clip_on=False, c='r', edgecolors='r',
    #                label='reliable infeasible', s=1, marker='s')
    ax.scatter(obs_e[..., 0], obs_e[..., 1], clip_on=False, c='g', edgecolors='g', label='demonstration', s=1)

    ax.set_ylim([y_lim[0], y_lim[1]])
    ax.set_xlim([x_lim[0], x_lim[1]])
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid('on')
    ax.set_title(fig_title, fontsize=15)
    plt.legend()
    fig.savefig(save_name, bbox_inches='tight')



def plot_for_UR5_envs_2D(cost_function, manual_threshold, position_limit, env, env_id, select_dim, obs_dim, acs_dim,
                      save_name, fig_title, obs_e, rew_e, len_e, obs_n, rew_n, len_n, n_policy, obs_f):
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    joint1_lim = [1, 3]
    joint2_lim = [-3, 0]

    r1 = np.linspace(joint1_lim[0], joint1_lim[1], num=100)
    r2 = np.linspace(joint2_lim[0], joint2_lim[1], num=100)
    X, Y = np.meshgrid(r1, r2)
    obs_all = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=-1)
    obs_all = np.concatenate((obs_all, np.ones((np.size(X), 3))*[-1.843, -1.294, 0.55]), axis=-1)   #[-1.92, -1.166, 0.641]
    obs_all = np.concatenate((obs_all, np.zeros((np.size(X), obs_dim-5))), axis=-1)
    plt.plot(2.26, -1.69, 'o', markersize=7, color='yellow')  #2.28, -1.55,
    # [[2.06, - 1.67,-1.659, -1.281, 0.8714]]

    action = np.zeros((np.size(X), acs_dim))
    outs = 1 - cost_function(obs_all, action)
    im = ax.imshow(outs.reshape(X.shape), extent=[joint1_lim[0], joint1_lim[1], joint2_lim[0], joint2_lim[1]],
                   cmap='jet_r', vmin=0, vmax=1, origin='lower')
    contour = plt.contour(X, Y, outs.reshape(X.shape), [1-manual_threshold], colors='k', linewidths=2.0)
    plt.clabel(contour, fontsize=15, colors=('k'))
    contour1 = ax.contour(X, Y, outs.reshape(X.shape), [0.5], colors='k', linewidths=2.0, linestyles='--')
    plt.clabel(contour1, fontsize=15, colors=('k'))
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=20)

    ax.set_ylim([joint2_lim[0], joint2_lim[1]])
    ax.set_xlim([joint1_lim[0], joint1_lim[1]])
    ax.set_xlabel('joint1', fontsize=20)
    ax.set_ylabel('joint2', fontsize=20)
    ax.tick_params(labelsize=20)
    plt.grid('on')
    plt.title(fig_title, fontsize=30)
    fig.savefig(save_name, bbox_inches='tight')
    plt.close(fig=fig)

def plot_for_UR5_envs_3D(cost_function, manual_threshold, acs_dim, save_name, fig_title, obs_n, len_n, n_traj_plot=10):
    plotter = pv.Plotter(off_screen=True)
    plotter.show_grid(color="lightgrey")
    plotter.add_axes()

    eval_data = np.load('icrl/expert_data/UR5WithPos/files/evaluation_states.npy')
    eval_obs = eval_data[:, :acs_dim]
    eval_eep = eval_data[:, 5:8]
    from icrl.utils_mujoco_UR5 import obs2pos
    plot_eep = obs2pos(obs_n) if obs_n is not None else []

    if cost_function is not None:
        true_constraint = eval_data[:, -1].astype(np.int64)  # 1 is infeasible
        obs_all = np.concatenate((eval_obs, eval_obs * 0), axis=-1)  # eval_obs*0 means zero speed
        action = np.zeros((eval_obs.shape[0], acs_dim))

        learned_constraint = cost_function(obs_all, action)
        learned_constraint = np.where(learned_constraint > manual_threshold, 1, 0).astype(np.int64)
        for i, point in enumerate(eval_eep):
            is_correct = learned_constraint[i] == true_constraint[i]
            color = 'red' if learned_constraint[i] == 1 else 'green'
            if not is_correct:
                plotter.add_mesh(pv.Sphere(radius=0.003, center=point), color=color)
            else:
                plotter.add_mesh(pv.Tetrahedron(center=point, radius=0.004), color=color)
            if i > 800: break
    else:
        # plot only expert data
        print('No cost function, plot expert data')

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'orange', 'purple']
    start_idx = 0
    n_traj_plot = min(len(len_n), n_traj_plot)
    for i, length in enumerate(len_n[:n_traj_plot]):
        end_idx = start_idx + length
        trajectory = plot_eep[start_idx:end_idx-1]
        points = trajectory  # Points of the trajectory
        lines = np.full((len(trajectory) - 1, 3), 2, dtype=np.int_)  # Each row: [2, point_id_start, point_id_end]
        lines[:, 1] = np.arange(0, len(trajectory) - 1)  # Start points of the line segments
        lines[:, 2] = np.arange(1, len(trajectory))  # End points of the line segments
        poly_data = pv.PolyData(points)
        poly_data.lines = lines
        plotter.add_mesh(poly_data, color=colors[i % len(colors)], line_width=3, render_lines_as_tubes=True)
        start_idx = end_idx

    center = [0, -0.6, 1.0]
    size = [0.08, 0.06, 0.5]  # Full size
    cube = pv.Cube(center=center, x_length=size[0], y_length=size[1], z_length=size[2])
    plotter.add_mesh(cube, color='gray', show_edges=True, edge_color='black', opacity=1)
    target = np.array([[0, -0.68, 1.05]])
    plotter.add_mesh(pv.Sphere(radius=0.02, center=target), color='purple')
    plotter.camera_position = [(0.2, -1.35, 1.8), (0, -0.68, 1.15), (0, 0, 1)]
    # plotter.show()
    plotter.screenshot(save_name)
    plotter.close()


    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # colors = ['green', 'red']
    # draw_cuboid(ax, [0, -0.6, 1.0], [0.04, 0.03, 0.25], 'gray')
    # for i in range(min(100, eval_obs.shape[0])):
    #     ax.scatter(eval_eep[i, 0], eval_eep[i, 1], eval_eep[i, 2], color=colors[learned_constraint[i]], s=1)
    # idx = 0
    # for len in len_n:
    #     trajectory = plot_eep[idx:idx + len - 1]
    #     ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    #     idx += len
    # # ax.tick_params(labelsize=20)
    # plt.grid('on')
    # ax.set_xlim([-0.5, 0.5])
    # ax.set_ylim([-0.7, 0.2])
    # ax.set_zlim([0.9, 1.5])
    # # plt.title(fig_title, fontsize=30)
    # plt.show()
    # # fig.savefig(save_name, bbox_inches='tight')
    # plt.close(fig=fig)

def plot_for_PointObs(cost_function, manual_threshold, obs_dim, acs_dim, save_name, fig_title, obs_n, obs_e, obs_rn, gmm_true=None, gmm_learned=None, query_obs=None, show=False):
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.4))
    x_lim = [-0.2, 1.3]
    y_lim = [-0.2, 1.6]

    r1 = np.linspace(x_lim[0], x_lim[1], num=200)
    r2 = np.linspace(y_lim[0], y_lim[1], num=200)
    X, Y = np.meshgrid(r1, r2)
    obs = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=-1)
    action = np.zeros((np.size(X), acs_dim))

    outs = 1 - cost_function(obs, action)
    im = ax.imshow(outs.reshape(X.shape), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]],
                   cmap='jet_r', vmin=0, vmax=1, origin='lower')
    contour = ax.contour(X, Y, outs.reshape(X.shape), [1 - manual_threshold], colors='k', linewidths=2.0)
    plt.clabel(contour, fontsize=8, colors=('k'))

    # cb = fig.colorbar(im, ax=ax)
    # cb.ax.tick_params(labelsize=10)

    ax.scatter(obs_n[..., 0], obs_n[..., 1], clip_on=False, c='y', edgecolors='y', label='policy $\mathcal{P}$', s=3)
    ax.scatter(obs_rn[..., 0], obs_rn[..., 1], clip_on=False, c='r', edgecolors='r', label='reliable infeasibles $\mathcal{R}\cup\mathcal{M}$', s=6)
    ax.scatter(obs_e[..., 0], obs_e[..., 1], clip_on=False, c='g', edgecolors='g', label='Feasible demonstration $\mathcal{D}$', s=6)
    if query_obs is not None:
        ax.scatter(query_obs[..., 0], query_obs[..., 1], clip_on=False, marker='*', label='query', s=80, c='white')

    if gmm_learned is not None:
        plot_gmm_ellipse(gmm_learned, ax, linestyle='--', label='fitted constraint')
    if gmm_true is not None:
        plot_gmm_ellipse(gmm_true, ax, edgecolor='w', linestyle='-', label='true constraint boundary')

    # policy.plot_vector_flow(ax, 0.15)

    ax.set_ylim([y_lim[0], y_lim[1]])
    ax.set_xlim([x_lim[0], x_lim[1]])
    # ax.set_xlabel('x', fontsize=10)
    # ax.set_ylabel('y', fontsize=10)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(MultipleLocator(0.3))
    ax.yaxis.set_major_locator(MultipleLocator(0.3))
    ax.grid('on')
    # ax.set_title(fig_title, fontsize=15)
    handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles[:4], labels[:4])

    fig.savefig(save_name, bbox_inches='tight', pad_inches=0.02,dpi=300)

def plot_for_ReachObs(cost_function, manual_threshold, obs_dim, acs_dim, cn_obs_select_dim, save_name, fig_title, obs_n, obs_e, len_e, len_n, obs_rn, gmm=None, **kwargs
                   ):

    if len(cn_obs_select_dim) == 2:
        fig, ax = plt.subplots(1, 1, figsize=(9, 8))

        x_lim = [0.3, .7]
        y_lim = [-0.2, 0.2]

        r1 = np.linspace(x_lim[0], x_lim[1], num=200)
        r2 = np.linspace(y_lim[0], y_lim[1], num=200)
        X, Y = np.meshgrid(r1, r2)

        obs = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1]), np.zeros((X.size, obs_dim-2))], axis=-1)
        action = np.zeros((np.size(X), acs_dim))

        outs = 1 - cost_function(obs, action)
        im = ax.imshow(outs.reshape(X.shape), extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]],
                       cmap='jet_r', vmin=0, vmax=1, origin='lower')
        contour = ax.contour(X, Y, outs.reshape(X.shape), [1 - manual_threshold], colors='k', linewidths=2.0)
        plt.clabel(contour, fontsize=8, colors=('k'))
        # contour1 = ax.contour(X, Y, outs.reshape(X.shape), [0.5], colors='k', linewidths=1.0, linestyles='--')
        # plt.clabel(contour1, fontsize=15, colors=('k'))
        # contour2 = ax.contour(X, Y, outs.reshape(X.shape), [0.01], colors='k', linewidths=1.0, linestyles='--' )
        # plt.clabel(contour2, fontsize=15, colors=('k'))
        # contour3 = ax.contour(X, Y, outs.reshape(X.shape), [0.1], colors='k', linewidths=1.0, linestyles='--')
        # plt.clabel(contour3, fontsize=15, colors=('k'))
        #
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(labelsize=10)

        # Plot the circles contained in the gmm
        if gmm is not None:
            for k in range(gmm.n_components):
                circle = plt.Circle(gmm.means_[k], (9e-4) ** 0.5, color='red', alpha=0.5)
                ax.add_patch(circle)

        ax.scatter(obs_n[..., 0], obs_n[..., 1], clip_on=False, c='orange', edgecolors='orange', label='policy', s=0.3)
        if obs_rn is not None:
            ax.scatter(obs_rn[..., 0], obs_rn[..., 1], clip_on=False, c='r', edgecolors='r', label='reliable infeasible', s=0.5, marker='s')
        ax.scatter(obs_e[..., 0], obs_e[..., 1], clip_on=False, c='g', edgecolors='g', label='demonstration', s=0.5)

        ax.set_ylim([y_lim[0], y_lim[1]])
        ax.set_xlim([x_lim[0], x_lim[1]])
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.tick_params(labelsize=10)
        ax.grid('on')
        ax.set_title(fig_title, fontsize=15)
        plt.legend()
        fig.savefig(save_name, bbox_inches='tight')


    if len(cn_obs_select_dim) == 3:
        x_lim = kwargs.get('x_lim', (0.3, 0.7))
        y_lim = kwargs.get('y_lim', (-0.15, 0.15))
        z_lim = kwargs.get('z_lim', (0.02, 0.25))
        num_points_for_plot = 50

        fig = plt.figure(figsize=(30, 13))
        if  (cost_function is not None):
            r1 = np.linspace(x_lim[0], x_lim[1], num=num_points_for_plot)
            r2 = np.linspace(y_lim[0], y_lim[1], num=num_points_for_plot)
            r3 = np.linspace(z_lim[0], z_lim[1], num=num_points_for_plot)
            X, Y, Z = np.meshgrid(r1, r2, r3)

            obs = np.concatenate(
                [X.reshape([-1, 1]), Y.reshape([-1, 1]), Z.reshape([-1, 1]), np.zeros((X.size, obs_dim - 3))], axis=-1)
            action = np.zeros((np.size(X), acs_dim))
            outs = (1 - cost_function(obs, action)).reshape(X.shape)

            ax1 = fig.add_subplot(121, projection='3d')

            colors = [(1, 0, 0, alpha) for alpha in np.linspace(0.9, 1, 100)]
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list('custom_reds', colors, N=100)
            mask = outs < 1 - manual_threshold
            X_masked, Y_masked, Z_masked, outs_masked = X[mask], Y[mask], Z[mask], outs[mask]
            ax1.scatter(X_masked, Y_masked, Z_masked, c=outs_masked, cmap=cmap, label='infeasible states', vmin=0, vmax=1.)

            if (obs_n is not None):
                ax1.scatter(obs_n[..., 0], obs_n[..., 1], obs_n[..., 2], clip_on=False, c='orange', edgecolors='orange',
                                label='policy', s=10)
            if (obs_e is not None):
                ax1.scatter(obs_e[..., 0], obs_e[..., 1], obs_e[..., 2], clip_on=False, c='g', edgecolors='g',
                            label='demonstration', s=10)
            ax1.set_xlim(x_lim)
            ax1.set_ylim(y_lim)
            ax1.set_zlim(z_lim)
            ax1.set_xlabel('x', fontsize=10)
            ax1.set_ylabel('y', fontsize=10)
            ax1.set_zlabel('z', fontsize=10)
            ax1.tick_params(labelsize=10)
            ax1.grid('on')
            ax1.set_title(f'{fig_title} - constrained states', fontsize=15)
            ax1.legend()

            def plot_projection(ax, x_data, y_data, c_data, obs_n, obs_e, x_idx, y_idx, x_label, y_label, x_lim, y_lim,
                                title, cmap):
                ax.scatter(x_data, y_data, c=c_data, cmap=cmap, label='infeasible states', vmin=0, vmax=0.5)

                if obs_n is not None:
                    ax.scatter(obs_n[..., x_idx], obs_n[..., y_idx], clip_on=False, c='orange', edgecolors='orange',
                               label='policy', s=10)

                if obs_e is not None:
                    ax.scatter(obs_e[..., x_idx], obs_e[..., y_idx], clip_on=False, c='g', edgecolors='g',
                               label='demonstration', s=10)

                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.set_xlabel(x_label, fontsize=10)
                ax.set_ylabel(y_label, fontsize=10)
                ax.tick_params(labelsize=10)
                ax.grid('on')
                ax.set_title(title, fontsize=15)

                ax.legend()

            fig2 = plt.figure(figsize=(18, 6))

            # plot projection on x-y, y-,z, x-z planes
            ax2 = fig2.add_subplot(131)
            plot_projection(ax2, X_masked, Y_masked, outs_masked, obs_n, obs_e, 0, 1, 'x', 'y', x_lim, y_lim,
                            'x-y Projection', cmap)

            # Plot the cylinder obstacle on x-y projection if the obstacle is defined using gmm
            if gmm is not None:
                for k in range(gmm.n_components):
                    circle = plt.Circle(gmm.means_[k], (9e-4) ** 0.5, color='yellow', alpha=0.5) #TODO: check the radius and optimize the gmm plotting
                    ax2.add_patch(circle)

            ax3 = fig2.add_subplot(132)
            plot_projection(ax3, Y_masked, Z_masked, outs_masked, obs_n, obs_e, 1, 2, 'y', 'z', y_lim, z_lim,
                            'y-z Projection', cmap)
            ax4 = fig2.add_subplot(133)
            plot_projection(ax4, X_masked, Z_masked, outs_masked, obs_n, obs_e, 0, 2, 'x', 'z', x_lim, z_lim,
                            'x-z Projection', cmap)
            fig2.savefig(save_name.replace(".png", "_2d.png"), bbox_inches='tight')

        ax5 = fig.add_subplot(122, projection='3d')
        # _, grad = grad_function(obs_n, np.zeros((obs_n.shape[0], acs_dim)))
        # ax5.quiver(obs_n[::30, 0], obs_n[::30, 1], obs_n[::30, 2], grad[::30, 0], grad[::30, 1], grad[::30, 2], color='black', length=0.001,
        #            label='Velocity Vectors')
        if obs_n is not None and obs_n.size != 0:
            ax5.scatter(obs_n[..., 0], obs_n[..., 1], obs_n[..., 2], clip_on=False, c='orange', edgecolors='orange', label='policy',
                        s=6)
        if obs_e is not None and obs_e.size != 0:
            ax5.scatter(obs_e[..., 0], obs_e[..., 1], obs_e[..., 2], clip_on=False, c='g', edgecolors='g',
                        label='demonstration', s=20)

        if obs_rn is not None and obs_rn.size != 0:
            ax5.scatter(obs_rn[..., 0], obs_rn[..., 1], obs_rn[..., 2], clip_on=False, c='r', edgecolors='r',
                        label='reliable infeasible states', s=20)
        if gmm is not None:
            for k in range(gmm.n_components):
                plot_cylinder(ax5, center=gmm.means_[k], radius=(9e-4)**0.5, height=0.08, color='blue', alpha=0.5)

        # Indicate the start and end of each trajectory and its index
        start_idx = 0
        for i, length in enumerate(len_e):
            end_idx = start_idx + length - 1
            start_point = obs_e[start_idx]
            end_point = obs_e[end_idx]
            ax5.text(start_point[0], start_point[1], start_point[2], f'Start {i}', color='green', fontsize=8)
            ax5.text(end_point[0], end_point[1], end_point[2], f'End {i}', color='red', fontsize=8)
            start_idx = end_idx + 1

        ax5.set_xlabel('x', fontsize=10)
        ax5.set_ylabel('y', fontsize=10)
        ax5.set_zlabel('z', fontsize=10)
        ax5.set_xlim(x_lim)
        ax5.set_ylim(y_lim)
        ax5.set_zlim(z_lim)
        ax5.tick_params(labelsize=10)
        ax5.grid('on')
        ax5.set_title(f'{fig_title} - Constraint learning', fontsize=15)
        ax5.legend()

        fig.savefig(save_name, bbox_inches='tight')


def plot_for_ReachVel(cost_function, manual_threshold, obs_dim, acs_dim, cn_obs_select_dim, save_name, fig_title, obs_n, obs_e, len_e, obs_rn):

    if len(cn_obs_select_dim) == 3:
        x_lim = [0.3, 0.7]
        y_lim = [-0.2, 0.2]
        z_lim = [0.02, 0.12]

        a_lim = [[0.5, 1.3],
                 [-1.3, 1.3],
                 [-1.3, 1.3]]
        fig = plt.figure(figsize=(30, 13))
        if  (cost_function is not None):
            r1 = np.linspace(a_lim[0][0], a_lim[0][1], num=15)
            r2 = np.linspace(a_lim[1][0], a_lim[1][1], num=15)
            r3 = np.linspace(a_lim[2][0], a_lim[2][1], num=15)
            X, Y, Z = np.meshgrid(r1, r2, r3)

            obs = np.concatenate(
                [np.zeros((X.size, obs_dim - 3)), X.reshape([-1, 1]), Y.reshape([-1, 1]), Z.reshape([-1, 1])], axis=-1)
            action = np.zeros((np.size(X), acs_dim))
            outs = (1 - cost_function(obs, action)).reshape(X.shape)

            # Plot 3D velocities distributions
            ax1 = fig.add_subplot(121, projection='3d')
            mask = outs > 1 - manual_threshold
            X_masked, Y_masked, Z_masked, outs_masked = X[mask], Y[mask], Z[mask], outs[mask]
            ax1.scatter(X_masked, Y_masked, Z_masked, c='y', label='Feasible')

            if (obs_n is not None):
                ax1.scatter(obs_n[..., 6], obs_n[..., 7], obs_n[..., 8], clip_on=False, c='orange', edgecolors='orange',
                                label='policy',
                                s=10)
            if (obs_e is not None):
                ax1.scatter(obs_e[..., 6], obs_e[..., 7], obs_e[..., 8], clip_on=False, c='g', edgecolors='g',
                            label='demonstration', s=10)
            if obs_rn is not None and obs_rn.size != 0:
                ax1.scatter(obs_rn[..., 6], obs_rn[..., 7], obs_rn[..., 8], clip_on=False, c='r', edgecolors='r',
                            label='reliable infeasible', s=6)
            ax1.set_xlim(a_lim[1])
            ax1.set_ylim(a_lim[1])
            ax1.set_zlim(a_lim[2])
            ax1.set_xlabel('vx', fontsize=10)
            ax1.set_ylabel('vy', fontsize=10)
            ax1.set_zlabel('vz', fontsize=10)
            ax1.tick_params(labelsize=10)
            ax1.grid('on')
            ax1.set_title(f'{fig_title} - Velocity distribution', fontsize=15)
            ax1.legend()

            def plot_2Dprojection(ax, x_data, y_data, obs_n, obs_e, obs_rn, x_idx, y_idx, x_label, y_label, a_lim_x,
                                a_lim_y, title):
                ax.scatter(x_data, y_data, c='y', label='Feasible')
                if obs_n is not None:
                    ax.scatter(obs_n[..., x_idx], obs_n[..., y_idx], clip_on=False, c='orange', edgecolors='orange',
                               label='policy', s=10)
                if obs_e is not None:
                    ax.scatter(obs_e[..., x_idx], obs_e[..., y_idx], clip_on=False, c='g', edgecolors='g',
                               label='demonstration', s=10)
                # if obs_rn is not None and obs_rn.size != 0:
                #     ax.scatter(obs_rn[..., x_idx], obs_rn[..., y_idx], clip_on=False, c='r', edgecolors='r',
                #                label='reliable infeasible', s=6)
                ax.set_xlim(a_lim_x)
                ax.set_ylim(a_lim_y)
                ax.set_xlabel(x_label, fontsize=10)
                ax.set_ylabel(y_label, fontsize=10)
                ax.tick_params(labelsize=10)
                ax.grid('on')
                ax.set_title(title, fontsize=15)
                ax.legend()

            # Plot the 2D projection of velocities
            fig2 = plt.figure(figsize=(18, 6))
            ax2 = fig2.add_subplot(141)
            plot_2Dprojection(ax2, X_masked, Y_masked, obs_n, obs_e, obs_rn, 6, 7, 'x', 'y', a_lim[0], a_lim[1],
                              'x-y Projection')
            ax3 = fig2.add_subplot(142)
            plot_2Dprojection(ax3, Y_masked, Z_masked, obs_n, obs_e, obs_rn, 7, 8, 'y', 'z', a_lim[1], a_lim[2],
                              'y-z Projection')
            ax4 = fig2.add_subplot(143)
            plot_2Dprojection(ax4, X_masked, Z_masked, obs_n, obs_e, obs_rn, 6, 8, 'x', 'z', a_lim[0], a_lim[2],
                              'x-z Projection')
            ax_pos = fig2.add_subplot(144)
            plot_2Dprojection(ax_pos, X_masked, Y_masked, obs_n, obs_e, obs_rn, 0, 1, 'x', 'y', x_lim, y_lim,
                              'x-y position Projection')
            fig2.savefig(save_name.replace(".png", "_2d.png"), bbox_inches='tight')

        # Plot the distribu of the velocities
        fig_hist = plt.figure(figsize=(5, 1.))

        ax_vx_hist = fig_hist.add_subplot(121)
        counts, bins, patches = ax_vx_hist.hist(obs_n[..., 6] * 0.48, bins=25, range=(0, 0.6), color='orange',
                                                edgecolor='black')
        proportions = counts / counts.sum()
        ax_vx_hist.cla()
        ax_vx_hist.bar(bins[:-1], proportions, width=np.diff(bins), color='orange', edgecolor='black', align='edge')
        ax_vx_hist.axvline(x=0.48, color='red', linestyle='--')
        # ax_vx_hist.set_xlabel(r'$v_x$', fontsize=10)
        # ax_vx_hist.legend(fontsize=7)
        ax_vx_hist.set_ylim(0, 0.6)
        ax_vx_hist.yaxis.set_major_locator(MaxNLocator(nbins=3))

        ax_vz_hist = fig_hist.add_subplot(122)
        counts, bins, patches = ax_vz_hist.hist(obs_n[..., 8] * 0.48, bins=25, range=(-0.3, 0.3), color='orange',
                                                edgecolor='black')
        proportions = counts / counts.sum()
        ax_vz_hist.cla()
        ax_vz_hist.bar(bins[:-1], proportions, width=np.diff(bins), color='orange', edgecolor='black', align='edge')
        ax_vz_hist.axvline(x=0.19, color='red', linestyle='--')
        ax_vz_hist.axvline(x=-0.19, color='red', linestyle='--')
        # ax_vz_hist.set_xlabel(r'$v_z$', fontsize=10)
        # ax_vz_hist.legend(fontsize=7)
        ax_vz_hist.set_ylim(0, 0.5)

        plt.subplots_adjust(wspace=0.1)
        fig_hist.tight_layout(pad=0.2)
        fig_hist.savefig(save_name.replace(".png", "_hist.png"), bbox_inches='tight', dpi=300, pad_inches=0.01)

        # Plot the 3D trajectories and obstacles
        ax5 = fig.add_subplot(122, projection='3d')
        if obs_n is not None and obs_n.size != 0:
            ax5.scatter(obs_n[..., 0], obs_n[..., 1], obs_n[..., 2], clip_on=False, c='orange', edgecolors='orange', label='policy',
                        s=6)
        if obs_e is not None and obs_e.size != 0:
            ax5.scatter(obs_e[..., 0], obs_e[..., 1], obs_e[..., 2], clip_on=False, c='g', edgecolors='g',
                        label='demonstration', s=20)
        # if obs_rn is not None and obs_rn.size != 0:
        #     ax5.scatter(obs_rn[..., 0], obs_rn[..., 1], obs_rn[..., 2], clip_on=False, c='r', edgecolors='r',
        #                 label='reliable infeasible', s=20)
        plot_cylinder(ax5, center=[0.55, 0, 0], radius=0.05, height=0.12, color='blue', alpha=0.5)

        ax5.set_xlabel('x', fontsize=10)
        ax5.set_ylabel('y', fontsize=10)
        ax5.set_zlabel('z', fontsize=10)
        ax5.set_xlim(x_lim)
        ax5.set_ylim(y_lim)
        ax5.set_zlim(z_lim)
        ax5.tick_params(labelsize=10)
        ax5.grid('on')
        ax5.set_title(f'{fig_title} - Observations', fontsize=15)
        ax5.legend()

        fig.savefig(save_name, bbox_inches='tight')

    else:
        raise NotImplementedError('The dimension of the observation is not supported')


def evaluate_constraint_accuracy(env_id, learned_cost_function, true_cost_function, obs_dim, acs_dim,):

        def collect_accuracy_metrics(label, pred):
            accuracy = (label == pred).mean()
            jaccard = np.sum(np.logical_and(label, pred)) / np.sum(
                np.logical_or(label, pred))
            false_positive_rate = (((label - pred) == -1).sum()) / (
                (label == 0).sum()) if (label == 0).sum() > 0 else 0
            recall = (np.logical_and(label == 1, pred == 1).sum()) / (
                (label == 1).sum()) if (label == 1).sum() > 0 else 0
            precision = (np.logical_and(label == 1, pred == 1).sum()) / (
                (pred == 1).sum()) if (pred == 1).sum() > 0 else 0
            F1score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
            accuracy_metrics = {'true/accuracy': accuracy,
                                'true/false_positive_rate': false_positive_rate,
                                'true/recall': recall,
                                'true/precision': precision,
                                'true/jaccard': jaccard,
                                'true/F1score': F1score
                                }
            return accuracy_metrics

        # For every environment, define the area where the accuracy is evaluated
        if env_id in ['UR5WithPos-v0', 'UR5WithPosTest-v0']:
            eval_data = np.load('icrl/expert_data/UR5WithPos/files/evaluation_states.npy')
            eval_obs = eval_data[:, :obs_dim//2]
            true_constraint = eval_data[:, -1].astype(np.int64) # 1 is infeasible
            obs = np.concatenate((eval_obs, eval_obs * 0), axis=-1) # eval_obs*0 means zero speed
            action = np.zeros((eval_obs.shape[0], acs_dim))
            learned_constraint = learned_cost_function(obs, action) # 1 is infeasible

        elif env_id in ['PointDS-v0', 'PointDSTest-v0', 'PointEllip-v0', 'PointEllipTest-v0']:
            r = np.arange(-0.1, 1.5, 0.03)
            X, Y = np.meshgrid(r, r)
            obs = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=-1)

            # # Only evaluate in a small region #TODO: need investigation
            # obs = obs[np.where(obs[:, 1] >= (obs[:, 0] * 0.4 / 1.8))]
            # obs = obs[np.where(obs[:, 1] <= (obs[:, 0] * 1.8 / 0.4))]

            action = np.zeros((obs.shape[0], acs_dim))
            true_constraint = true_cost_function(obs, action)  # 1 is infeasible
            learned_constraint = learned_cost_function(obs, action)

        elif env_id in ['ReachObs-v0']:
            x_lim = [0.45, 0.65]
            y_lim = [-0.1, 0.1]
            z_lim = [0.035, 0.08]

            r1 = np.linspace(x_lim[0], x_lim[1], num=40)
            r2 = np.linspace(y_lim[0], y_lim[1], num=40)
            r3 = np.linspace(z_lim[0], z_lim[1], num=40)
            X, Y, Z = np.meshgrid(r1, r2, r3)

            obs = np.concatenate(
                [X.reshape([-1, 1]), Y.reshape([-1, 1]), Z.reshape([-1, 1]), np.zeros((X.size, obs_dim - 3))],
                axis=-1)

            action = np.zeros((obs.shape[0], acs_dim))
            true_constraint = true_cost_function(obs, action)  # 1 is infeasible
            learned_constraint = learned_cost_function(obs, action)

        elif env_id in ['ReachConcaveObs-v0']:
            x_lim = [0.41, 0.6]
            y_lim = [-0.1, 0.18] if 'concave' in env_id else [-0.18, 0.18]
            z_lim = [0.035, 0.35]

            r1 = np.linspace(x_lim[0], x_lim[1], num=50)
            r2 = np.linspace(y_lim[0], y_lim[1], num=50)
            r3 = np.linspace(z_lim[0], z_lim[1], num=20)
            X, Y, Z = np.meshgrid(r1, r2, r3)

            obs = np.concatenate(
                [X.reshape([-1, 1]), Y.reshape([-1, 1]), Z.reshape([-1, 1]), np.zeros((X.size, obs_dim - 3))],
                axis=-1)

            action = np.zeros((obs.shape[0], acs_dim))
            true_constraint = true_cost_function(obs, action)  # 1 is infeasible
            learned_constraint = learned_cost_function(obs, action)

        elif env_id in ['ReachVelObs-v0', 'ReachVel-v0']:
            a_lim = [[0.7, 1.3],
                     [-1.3, 1.3],
                     [-1.3, 1.3]]
            r1 = np.linspace(a_lim[0][0], a_lim[0][1], num=40)
            r2 = np.linspace(a_lim[1][0], a_lim[1][1], num=60)
            r3 = np.linspace(a_lim[2][0], a_lim[2][1], num=60)
            X, Y, Z = np.meshgrid(r1, r2, r3)

            obs = np.concatenate(
                [np.zeros((X.size, obs_dim - 3)), X.reshape([-1, 1]), Y.reshape([-1, 1]), Z.reshape([-1, 1])],
                axis=-1)
            action = np.zeros((np.size(X), acs_dim))
            true_constraint = true_cost_function(obs, action)  # 1 is infeasible
            learned_constraint = learned_cost_function(obs, action)

        elif env_id in ['PointCircle-v0', 'PointCircleTest-v0']:
            r = np.arange(-13, 13, 0.1)
            X, Y = np.meshgrid(r, r)
            obs = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=-1)
            obs = np.concatenate((obs, np.zeros((np.size(X), obs_dim - 2))), axis=-1)
            action = np.zeros((np.size(X), acs_dim))
            true_constraint = true_cost_function(obs, action) # 1 is infeasible
            learned_constraint = learned_cost_function(obs, action)
        elif env_id in ['PointObstacleTest-v0', 'PointObstacle-v0', 'PointObstacle2-v0', 'PointObstacle2Test-v0']:
            r = np.arange(-7, 5, 0.1)
            X, Y = np.meshgrid(np.arange(-1, 6, 0.05), r)
            obs = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=-1)
            obs = np.concatenate((obs, np.zeros((np.size(X), obs_dim - 2))), axis=-1)
            action = np.zeros((np.size(X), acs_dim))
            true_constraint = true_cost_function(obs, action)  # 1 is infeasible
            learned_constraint = learned_cost_function(obs, action)
        elif env_id in ['HCWithPos-v0', 'HCWithPosTest-v0']:
            print('Warning! Halfcheetach env evaluate accuracy on fixed dataset. Make sure the evaluation states are correctly generated before')
            eval_data = np.load('icrl/expert_data/HCWithPos-New/files/evaluation_states.npy')
            obs = eval_data[:, :obs_dim]
            action = eval_data[:, obs_dim:obs_dim + acs_dim]
            true_constraint = eval_data[:, -1].astype(np.int64)  # 1 is infeasible
            learned_constraint = learned_cost_function(obs, action)

            # x_lim = [-12, 12]
            # y_lim = [-0.5, 0.5]
            # r1 = np.arange(x_lim[0], x_lim[1], 0.1)
            # r2 = np.arange(y_lim[0], y_lim[1], 0.1)
            # X, Y = np.meshgrid(r1, r2)
            # obs = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=-1)
            # obs = np.concatenate((obs, np.zeros((np.size(X), obs_dim - 2))), axis=-1)
            # action = np.zeros((np.size(X), acs_dim))
            # true_constraint = true_cost_function(obs, action)  # 1 is infeasible
            # learned_constraint = learned_cost_function(obs, action)
        else:
            raise NotImplementedError('The evaluation of this environment is not defined')

        return collect_accuracy_metrics(true_constraint, learned_constraint)

def evaluate_policy_accuracy(nominal_agent, expert_obs, expert_acs, expert_len, nominal_obs, nominal_len):
    # This function compares the accuracy of the policy with the expert demonstrations.
    nominal_acs, _ = nominal_agent(expert_obs)
    MSE = ((((nominal_acs - expert_acs) ** 2).sum(1))**0.5).mean()

    import fastdtw
    from scipy.spatial.distance import euclidean
    dtw_errors = []
    nominal_start = 0
    expert_start = 0

    for nom_len, exp_len in zip(nominal_len, expert_len):
        nom_traj = nominal_obs[nominal_start:nominal_start + nom_len, :2]
        exp_traj = expert_obs[expert_start:expert_start + exp_len, :2]

        distance, _ = fastdtw.fastdtw(nom_traj, exp_traj, dist=euclidean)
        dtw_errors.append(distance)
        nominal_start += nom_len
        expert_start += exp_len
    average_dtw_error = np.mean(dtw_errors)

    return {'true/policy_expert_MSE': MSE, 'true/policy_expert_dtwd':average_dtw_error}

def synthesis_query(pos_data, neg_data, n_synthesis_data, n_query_data, cost_function):
    # This function is only used for active learning
    # Generate query samples for active learning setting
    # r = np.arange(-0.1, 1.6, 0.03)
    # X, Y = np.meshgrid(r, r)
    # synthesis_candidates = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=-1)
    kmeans = KMeans(n_clusters=30)
    kmeans.fit(pos_data)
    pos_data_center = kmeans.cluster_centers_
    indices_pos = np.random.randint(0, pos_data_center.shape[0], size=n_synthesis_data)
    if neg_data.shape[0] == 0:
        print('Error! No reliable infeasible data, please adjust the PU learning threshold')
    indices_neg = np.random.randint(0, neg_data.shape[0], size=n_synthesis_data)
    weight = np.random.uniform(0.3,0.7, [n_synthesis_data,1])
    synthesis_candidates = (weight*pos_data_center[indices_pos] + (1-weight) * neg_data[indices_neg])

    # indices_query = np.random.randint(0, synthesis_candidates.shape[0], size=n_query_data)
    # query_data = synthesis_candidates[indices_query]

    # synthesis_candidates = []
    # nn_pos = NearestNeighbors(n_neighbors=1)
    # nn_neg = NearestNeighbors(n_neighbors=3)
    # nn_pos.fit(pos_data)
    # nn_neg.fit(neg_data)
    # for neg_point in neg_data:
    #     pos_point_idx = nn_pos.kneighbors(neg_point[np.newaxis,:], return_distance=False)
    #     pos_point = pos_data[pos_point_idx[0,0]]
    #     synthesis_candidates.append(0.5*pos_point+0.5*neg_point)
    # synthesis_candidates = np.vstack(synthesis_candidates)

    # for _ in range(1):
    #     nn_pos.fit(pos_data)
    #     nn_neg.fit(neg_data)
    #     min_diff = np.inf
    #
    #     for candidate in synthesis_candidates:
    #         distances_pos, _ = nn_pos.kneighbors([candidate], return_distance=True)
    #         distances_neg, _ = nn_neg.kneighbors([candidate], return_distance=True)
    #         diff = abs(distances_pos - distances_neg)
    #         if diff <= 0.05 : # or (distances_neg>0.15 and distances_pos >0.15)
    #             max_uncertain_data = candidate[np.newaxis, :]
    #             query_data.append(max_uncertain_data)
    #     query_data = np.vstack(query_data)
    #     query_data = query_data[np.random.choice(query_data.shape[0], 3, replace=True)]

    # -------------------------------------------
    # Query by nearest neighbour
    # -------------------------------------------
    # nn_pos = NearestNeighbors(n_neighbors=1)
    # nn_neg = NearestNeighbors(n_neighbors=1)
    # query_data = []
    # for _ in range(1):
    #     nn_pos.fit(pos_data)
    #     nn_neg.fit(neg_data)
    #     min_diff = np.inf
    #
    #     for candidate in synthesis_candidates:
    #         distances_pos, _ = nn_pos.kneighbors([candidate], return_distance=True)
    #         distances_neg, _ = nn_neg.kneighbors([candidate], return_distance=True)
    #         diff = abs(distances_pos - distances_neg)
    #         if diff <= 0.05 : # or (distances_neg>0.15 and distances_pos >0.15)
    #             max_uncertain_data = candidate[np.newaxis, :]
    #             query_data.append(max_uncertain_data)
    #     query_data = np.vstack(query_data)
    #     query_data = query_data[np.random.choice(query_data.shape[0], n_query_data, replace=True)]

    # -------------------------------------------
    # No query
    # -------------------------------------------

    # query_data = np.random.uniform(low=0, high=1.6, size=(0, 2))

    # -------------------------------------------
    # Query by kNN number of data
    # ------------------------------------------
    # def create_active_learning_data(synthesis_candidates, combine_data, labels, k_kNN=5, max_diff=1):
    #     nn = NearestNeighbors(n_neighbors=k_kNN)
    #     nn.fit(combine_data)
    #
    #     active_learning_data = []
    #
    #     for candidate in synthesis_candidates:
    #         distances, indices = nn.kneighbors([candidate], return_distance=True)
    #         neighbor_labels = labels[indices.flatten()]
    #         pos_count = np.sum(neighbor_labels == 1)
    #         neg_count = np.sum(neighbor_labels == 0)
    #         diff = abs(pos_count - neg_count)
    #         if diff <= max_diff:
    #             active_learning_data.append(candidate)
    #
    #     active_learning_data = np.array(active_learning_data)
    #     # kmeans = KMeans(n_clusters=n_query_data)
    #     # kmeans.fit(active_learning_data)
    #     #
    #
    #     # cluster_centers = kmeans.cluster_centers_
    #
    #     return active_learning_data
    # combined_data = np.vstack([pos_data, neg_data])
    # nominal_labels = np.zeros((neg_data.shape[0], 1))
    # expert_labels = np.ones((pos_data.shape[0], 1))
    # combined_labels = np.vstack((expert_labels, nominal_labels))
    # query_data = create_active_learning_data(synthesis_candidates, combined_data, combined_labels)

    # -------------------------------------------
    # Query by NN output uncertainty
    # -------------------------------------------
    # preds = cost_function(synthesis_candidates, np.zeros((synthesis_candidates.shape[0], 2)))
    # query_indices = np.argsort(np.abs(preds - 0.5))[:n_query_data]
    # query_data = synthesis_candidates[query_indices]
    #
    # kmeans = KMeans(n_clusters=n_query_data)
    # kmeans.fit(query_data)
    # query_data = kmeans.cluster_centers_

    # -------------------------------------------
    # Query by uncertainty - redundancy + diversity
    # -------------------------------------------
    combined_data = np.vstack([pos_data, neg_data])
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(combined_data)
    query_data = np.zeros([0, pos_data.shape[1]])
    q_nn = NearestNeighbors(n_neighbors=1)

    for i in range(n_query_data):
        preds = cost_function(synthesis_candidates, np.zeros((synthesis_candidates.shape[0], 2)))
        entropy = - preds * np.log(preds + 1e-5) - (1 - preds) * np.log(1 - preds + 1e-5)
        diversity = nn.kneighbors(synthesis_candidates, return_distance=True)[0].squeeze()
        if i > 0:
            redundancy = -q_nn.kneighbors(synthesis_candidates, return_distance=True)[0].squeeze()
        else:
            redundancy = np.zeros([synthesis_candidates.shape[0]])

        if i == n_query_data :
            scores = - redundancy * 0.2 + diversity
        else:
            scores = entropy - redundancy

        query_indices = np.argsort(scores)[-1]
        query_data = np.vstack([query_data, synthesis_candidates[query_indices]])

        q_nn.fit(query_data)

    # -------------------------------------------
    # Query by KDE
    # -------------------------------------------

    # combined_data = np.vstack([pos_data, neg_data])
    # data_width = np.std(combined_data, axis=0)
    # kde = KernelDensity(bandwidth=np.mean(data_width) / 8, kernel='gaussian')
    # for i in range(n_query_data):
    #     kde.fit(combined_data)
    #     density_scores = kde.score_samples(synthesis_candidates)
    #
    #     lowest_density_indices = density_scores.argsort()[:1]
    #     query_data = synthesis_candidates[lowest_density_indices]
    #     combined_data = np.vstack([combined_data, query_data])
    #
    # query_data = combined_data[-n_query_data:]

    # -------------------------------------------
    # Query based on close positive and negative KDE
    # -------------------------------------------

    # combined_data = np.vstack([pos_data, neg_data])
    # data_width = np.std(combined_data, axis=0)
    # kde_pos = KernelDensity(bandwidth=np.mean(data_width) / 4, kernel='gaussian')
    # kde_neg = KernelDensity(bandwidth=np.mean(data_width) / 4, kernel='gaussian')
    # for i in range(n_query_data):
    #     kde_pos.fit(pos_data)
    #     kde_neg.fit(neg_data)
    #     density_scores_pos = np.exp(kde_pos.score_samples(synthesis_candidates))
    #     density_scores_neg = np.exp(kde_neg.score_samples(synthesis_candidates))
    #     # lowest_density_indices = np.abs(density_scores_pos-density_scores_neg).argsort()[:1]
    #     lowest_density_indices = np.abs(density_scores_pos+density_scores_neg).argsort()[:1]
    #     query_point = synthesis_candidates[lowest_density_indices]
    #     pos_data = np.concatenate((pos_data, query_point))
    #     neg_data = np.concatenate((neg_data, query_point))
    #     combined_data = np.vstack([combined_data, query_point])
    #
    # query_data = combined_data[-n_query_data:]

    # -------------------------------------------
    # Plotting KDE distribution
    # matplotlib.use('TkAgg')
    # fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    # x_min, x_max = combined_data[:, 0].min() - 1, combined_data[:, 0].max() + 1
    # y_min, y_max = combined_data[:, 1].min() - 1, combined_data[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
    #                      np.linspace(y_min, y_max, 200))
    # # Z = np.exp(kde_pos.score_samples(np.c_[xx.ravel(), yy.ravel()])) + np.exp(kde_neg.score_samples(np.c_[xx.ravel(), yy.ravel()]))
    # Z = np.exp(kde.score_samples(np.c_[xx.ravel(), yy.ravel()]))
    # Z = Z.reshape(xx.shape)
    #
    # # plt.contourf(xx, yy, Z,  alpha=0.8)
    # ax.imshow(Z.reshape(xx.shape), extent=[x_min, x_max, y_min, y_max],
    #            cmap='jet_r', vmin=0, vmax=3, origin='lower')
    # ax.scatter(combined_data[:, 0], combined_data[:, 1], c='blue', s=20,)
    # ax.scatter(synthesis_candidates[:, 0], synthesis_candidates[:, 1], c='g', s=20)
    # ax.scatter(query_data[:, 0], query_data[:, 1], c='w', marker='*', s=50, edgecolor='w')
    #
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Kernel Density Estimation and Scatter Plot')
    # plt.show()
    return query_data
