import argparse
import copy
import os
import pickle
import sys
import time
import gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy, evaluate_policy_with_cost
from stable_baselines3.common.vec_env import sync_envs_normalization, VecNormalize
from tqdm import tqdm
import icrl.utils as utils
import wandb
from icrl.constraint_net import ConstraintNet, plot_for_PointObs, plot_constraints, evaluate_constraint_accuracy, evaluate_policy_accuracy
from icrl.true_constraint_net import get_true_cost_function, null_cost
from icrl.ds_policy import DS_Policy

def dscl(config):
    # Initialize DS agent and specify the target point
    nominal_agent = None
    if 'ReachConcave' in config.train_env_id:
        A = np.diag([-2, -2, -2])
        linear_ds = lambda x: A @ (x[:3] - np.array([0.68, 0, 0.04]))
        nominal_agent = DS_Policy(3, linear_ds)
    elif 'Reach2regions' in config.train_env_id:
        A = np.diag([-2, -2, -2])
        linear_ds = lambda x: A @ (x[:3] - np.array([0.62, 0.03, 0.04]))
        nominal_agent = DS_Policy(3, linear_ds)
    elif 'ReachObs' in config.train_env_id:
        A = np.diag([-2, -2, -2])
        linear_ds = lambda x: A @ (x[:3] - np.array([0.68, 0, 0.04]))
        nominal_agent = DS_Policy(3, linear_ds)
    elif 'Point' in config.train_env_id:
        A = np.diag([-1, -1])
        linear_ds = lambda x: A @ (x[:2])
        nominal_agent = DS_Policy(2, linear_ds)


    # the cost wrapper is only used for custom environments.
    use_cost_wrapper_train = True
    use_cost_wrapper_eval = False

    # creating the vectorized environments
    train_env = utils.make_train_env(env_id=config.train_env_id,
                                     save_dir=config.save_dir,
                                     use_cost_wrapper=use_cost_wrapper_train,
                                     base_seed=config.seed,
                                     num_threads=config.num_threads,
                                     normalize_obs=not config.dont_normalize_obs,
                                     normalize_reward=not config.dont_normalize_reward,
                                     normalize_cost=not config.dont_normalize_cost,
                                     cost_info_str=config.cost_info_str,
                                     reward_gamma=config.reward_gamma,
                                     cost_gamma=config.cost_gamma)

    # We don't need cost when taking samples
    sampling_env = utils.make_eval_env(env_id=config.train_env_id,
                                       use_cost_wrapper=False,  # cost is not needed when taking samples
                                       normalize_obs=not config.dont_normalize_obs)

    eval_env = utils.make_eval_env(env_id=config.eval_env_id,
                                   use_cost_wrapper=use_cost_wrapper_eval,
                                   normalize_obs=not config.dont_normalize_obs)

    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.shape[0]
    action_low, action_high = None, None
    if isinstance(sampling_env.action_space, gym.spaces.Box):
        action_low, action_high = sampling_env.action_space.low, sampling_env.action_space.high

    # loading the expert data
    (expert_obs, expert_acs, expert_reward), expert_lengths, expert_mean_reward = utils.load_expert_data(
        config.expert_path, config.expert_rollouts)
    # Set the eval starting point to be the same as expert data for fair comparison.
    start_indices = np.cumsum(np.insert(expert_lengths[:-1], 0, 0))
    starting_points = expert_obs[start_indices]
    eval_env.env_method('set_starting_point', starting_points)

    # initializing the constraint net
    cn_lr_schedule = lambda x: (config.anneal_clr_by_factor**(config.n_iters*(1-x))) * config.cn_learning_rate
    constraint_net = ConstraintNet(obs_dim, acs_dim, config.cn_layers, config.cn_batch_size, cn_lr_schedule,
                                   is_discrete, config.cn_reg_coeff, config.cn_obs_select_dim, config.cn_acs_select_dim,
                                   no_importance_sampling=config.no_importance_sampling,
                                   per_step_importance_sampling=config.per_step_importance_sampling,
                                   clip_obs=config.clip_obs,
                                   initial_obs_mean=None if not config.cn_normalize else np.zeros(obs_dim),
                                   initial_obs_var=None if not config.cn_normalize else np.ones(obs_dim),
                                   action_low=action_low, action_high=action_high,
                                   target_kl_old_new=config.cn_target_kl_old_new,
                                   target_kl_new_old=config.cn_target_kl_new_old,
                                   train_gail_lambda=config.train_gail_lambda, eps=config.cn_eps, device=config.device,
                                   loss_type=config.cn_loss_type, manual_threshold=config.cn_manual_threshold,
                                   weight_expert_loss=1, weight_nominal_loss=1,
                                   initial_feasible_bias=config.cn_initial_feasible_bias)
    # wandb.watch(constraint_net.network, log='all', log_freq=3000)

    if config.loading_constraint:
        constraint_net._load(f'icrl/wandb/{config.loading_constraint_dir}/files/models/icrl_{config.loading_constraint_ite}_itrs/cn.pt')

    # passing the cost function to the cost wrapper
    true_cost_function = get_true_cost_function(config.eval_env_id)
    if config.use_true_constraint:
        train_env.set_cost_function(true_cost_function)
        plot_cost_function = true_cost_function
    else:
        train_env.set_cost_function(constraint_net.cost_function)
        plot_cost_function = constraint_net.cost_function_non_binary

    cn_plot_dir = os.path.join(config.save_dir, 'constraint_net')
    utils.del_and_make(cn_plot_dir)

    timesteps = 0
    start_time = time.time()

    # training
    print(utils.colorize('\ntraining', color='green', bold=True), flush=True)
    best_true_reward, best_true_cost, best_recall, best_jaccard = -np.inf, np.inf, -np.inf, -np.inf

    # Initialize memory
    nominal_obs_memory = np.zeros([0, obs_dim])
    nominal_acs_memory = np.zeros([0, acs_dim])
    nominal_len_memory = np.zeros([0])
    rn_nominal_obs = None

    for itr in range(config.n_iters + 1): # Plus 1 because the last iteration is only used for updating policy and record final results
        # --------------------------------
        # Forward step, update policy
        if itr > 0: # skip the first iteration since the initial constraint network is usually terrible
            constraint_net_static = copy.deepcopy(constraint_net) # the policy is always with respect to the last constraint network.
            if config.ds_modulation_with_refer_point:
                nominal_agent.modulate_with_NN_with_refer_point(constraint_net_static.build_gamma_with_grad_for_ds, config.ds_modulation_rho)
            else:
                nominal_agent.modulate_with_NN(constraint_net_static.build_gamma_with_grad_for_ds, config.ds_modulation_rho)

        # --------------------------------
        # Backward step, update constraint
        # sampling nominal trajectories (from agent not expert)
        sync_envs_normalization(train_env, sampling_env)

        nominal_obs, _, nominal_acs, nominal_rew, nominal_len = utils.sample_from_same_starting_points_as_demonstrations(
            nominal_agent, sampling_env, expert_obs, expert_lengths, config.rollouts_per_demonstration,
            deterministic=True, policy_excerpts=config.select_policy_excerpts, expert_reward=expert_reward,
            cost_function=constraint_net.cost_function_non_binary, policy_excerpts_weight=config.policy_excerpts_weight)

        # update memory buffer
        if config.train_with_memory:
            nominal_obs_memory = np.concatenate((nominal_obs_memory, nominal_obs), 0)
            nominal_acs_memory = np.concatenate((nominal_acs_memory, nominal_acs), 0)
            nominal_len_memory = np.concatenate((nominal_len_memory, nominal_len))
        else:
            nominal_obs_memory = nominal_obs
            nominal_acs_memory = nominal_acs
            nominal_len_memory = nominal_len

        backward_metrics = []
        if itr < config.n_iters: # In the last extra ite, we only update policy and evaluate policy.

            # updating the constraint net (backward iterations)

            # If specified, train cn with true data, which is used for generating expert demonstrations by running ds_policy.py
            if config.train_constraint_with_true_data:
                feasible_obs, feasible_acs, infeasible_obs, infeasible_acs = utils.sample_true_labeled_data(true_cost_function, config.train_env_id)
                backward_metrics = constraint_net.train_MECL_BC(config.backward_iters, feasible_obs, feasible_acs,
                                                                infeasible_obs, infeasible_acs, None,
                                                                config.WB_label_frequency)

            elif config.cn_training_mode == 'mil':
                if nominal_len_memory.size == 0:
                    rn_nominal_obs = nominal_obs_memory
                    backward_metrics = {
                        'backward/cn_loss': np.nan,
                        'backward/mil_skipped_no_nominal_traj': 1,
                    }
                else:
                    _, rn_nominal_obs, backward_metrics = constraint_net.train_with_MIL(
                        config.backward_iters,
                        expert_obs,
                        expert_acs,
                        expert_lengths,
                        nominal_obs_memory,
                        nominal_acs_memory,
                        nominal_len_memory,
                        config,
                    )
                print('MIL backward step constraint loss:', backward_metrics.get('backward/cn_loss'))

            else:
                _, rn_nominal_obs, backward_metrics = constraint_net.train_with_two_step_pu_learning(
                    config.backward_iters, expert_obs, expert_acs, nominal_obs_memory, nominal_acs_memory,
                    nominal_len_memory, config)
        # --------------------------------
        # Save:
        # plotting
        if (itr % config.cn_plot_every == 0) or (itr == config.n_iters):
            plot_constraints(plot_cost_function, config.cn_manual_threshold, None, None, config.eval_env_id,
                             config.cn_obs_select_dim, obs_dim, acs_dim,
                             os.path.join(cn_plot_dir, '%d.png' % (itr + 1)), f'DSCL iter {itr + 1}', expert_obs, None,
                             expert_lengths, obs_nominal=nominal_obs, rew_nominal=None, len_nominal=nominal_len,
                             policy_excerpt_rollout_num=None, obs_failed=None, obs_rn=rn_nominal_obs, gmm_learned=None,
                             query_obs=None, show=config.show_figures,
                             grad_function=constraint_net.build_gamma_with_grad_for_ds)

        # Periodically save model, evaluate model
        if itr % config.eval_every == 0 or itr == config.n_iters:
            # evaluating the reward in the true environment
            sync_envs_normalization(train_env, eval_env)
            average_true_feasible_reward, average_true_reward, average_true_cost, true_safe_portion, average_episode_length = evaluate_policy_with_cost(
                nominal_agent, eval_env, true_cost_function, n_eval_episodes=config.evaluation_episode_num,
                deterministic=False)

            # evaluating the accuracy of constraint net
            accuracy_metrics = evaluate_constraint_accuracy(config.train_env_id,
                                                            constraint_net.cost_function_evaluation, true_cost_function,
                                                            obs_dim, acs_dim)
            # # evaluating the accuracy of policy
            # test_obs, _, _, _, test_len = utils.sample_from_same_starting_points_as_demonstrations(nominal_agent,
            #                                                                                        eval_env, expert_obs,
            #                                                                                        expert_lengths, 1,
            #                                                                                        deterministic=True,
            #                                                                                        cost_function=constraint_net.cost_function_non_binary)
            # accuracy_metrics.update(evaluate_policy_accuracy(nominal_agent, expert_obs, expert_acs, expert_lengths, test_obs, test_len))

        if itr % config.save_every == 0 or itr == config.n_iters:
            path = os.path.join(config.save_dir, f"models/icrl_{itr+1}_itrs")
            utils.del_and_make(path)
            # nominal_agent.save(os.path.join(path, f'nominal_agent'))
            constraint_net.save(os.path.join(path, f'cn.pt'))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(path, f'{itr+1}_train_env_stats.pkl'))

        # Update best index
        if average_true_cost < best_true_cost:
            best_true_cost = average_true_cost
        if accuracy_metrics['true/recall'] > best_recall:
            best_recall = accuracy_metrics['true/recall']
        if accuracy_metrics['true/jaccard'] > best_jaccard:
            best_jaccard = accuracy_metrics['true/jaccard']

        if itr % config.eval_every == 0 or itr == config.n_iters - 1:
            # Collect metrics
            metrics = {
                    "time(m)": (time.time()-start_time)/60,
                    "iteration": itr,
                    "true/feasible_reward": average_true_feasible_reward,
                    "true/average_reward": average_true_reward,
                    "true/violation_episodes_portion": 1 - true_safe_portion,
                    "true/violation_steps_portion": average_true_cost,
                    "true/average_episode_length": average_episode_length,
                    # "best_true/best_reward": best_true_reward,
                    "best_true/best_cost": best_true_cost,
                    "best_true/best_recall": best_recall,
                    "best_true/best_jaccard": best_jaccard
                    }
            metrics.update(backward_metrics)
            metrics.update(accuracy_metrics)
            if config.use_wandb:
                wandb.log(metrics)


    # making video of the final model
    if not config.wandb_sweep and config.use_wandb:
        sync_envs_normalization(train_env, eval_env)
        constraint_images = []
        for filename in os.listdir(cn_plot_dir):
            if filename.endswith('.png'):
                path = os.path.join(cn_plot_dir, filename)
                if filename[1].isdigit():
                    itr = int(filename[:2])
                else:
                    itr = int(filename[0])
                constraint_images.append(wandb.Image(path, caption=f'Ite {itr}'))
        wandb.log({'constraint_images': constraint_images})
        # utils.eval_and_make_video(eval_env, nominal_agent, config.save_dir, 'final_policy')

    if config.sync_wandb and config.use_wandb:
        utils.sync_wandb(config.save_dir, 120)


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    # loading_policy_ite has been removed by Erfan.
    # ========================== setup ============================== #
    parser.add_argument('file_to_run', type=str)
    parser.add_argument('--config_file', '-cf', type=str, default=None)
    parser.add_argument('--project', '-p', type=str, default='ICRL-FE2')
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--group', '-g', type=str, default='Point-ICRL')
    parser.add_argument('--device', '-d', type=str, default='cpu')
    parser.add_argument('--verbose', '-v', type=int, default=0) # default: 2
    parser.add_argument('--sync_wandb', '-sw', action='store_true')
    parser.add_argument('--wandb_sweep', '-ws', type=bool, default=False)
    parser.add_argument('--debug_mode', '-dm', action='store_true')
    # parser.add_argument('--loading_policy', '-lp', action='store_true')
    # parser.add_argument('--loading_policy_dir', '-lpd', type=str, default='run-20230420_175413-x4339028') # run-20230404_094232-vkzsib2p
    # parser.add_argument('--loading_policy_ite', '-lpi', type=str, default='-1')
    parser.add_argument('--loading_constraint', '-lc', action='store_true')
    parser.add_argument('--loading_constraint_dir', '-lcd', type=str, default=None)
    parser.add_argument('--loading_constraint_ite', '-lci', type=str, default=None)
    parser.add_argument('--table_function', '-tf', action='store_true')
    # ======================== environments ========================= #
    parser.add_argument('--train_env_id', '-tei', type=str, default='PointCircle-v0')
    parser.add_argument('--eval_env_id', '-eei', type=str, default='PointCircleTest-v0')
    parser.add_argument('--dont_normalize_obs', '-dno', action='store_true')
    parser.add_argument('--dont_normalize_reward', '-dnr', action='store_true')
    parser.add_argument('--dont_normalize_cost', '-dnc', action='store_true')
    parser.add_argument('--seed', '-s', type=int, default=23) # default: 23
    parser.add_argument('--clip_obs', '-co', type=int, default=20)
    # ============================ cost ============================= #
    parser.add_argument('--cost_info_str', '-cis', type=str, default='cost')

    # ====================== DS policy  ======================== #
    parser.add_argument('--ds_modulation_rho', '-dmr', type=float, default=0.2)
    parser.add_argument('--ds_modulation_with_refer_point', '-dmwr', action='store_true')
    parser.add_argument('--num_threads', '-nt', type=int, default=1)
    parser.add_argument('--train_constraint_with_true_data', '-tcwd', action='store_true')
    parser.add_argument('--save_every', '-se', type=float, default=1)
    parser.add_argument('--eval_every', '-ee', type=float, default=1)
    # ======================== policy mdp =========================== #
    parser.add_argument('--reward_gamma', '-rg', type=float, default=0.99)
    parser.add_argument('--reward_gae_lambda', '-rgl', type=float, default=0.95)
    parser.add_argument('--cost_gamma', '-cg', type=float, default=0.99)
    parser.add_argument('--cost_gae_lambda', '-cgl', type=float, default=0.95)
    # ========================== DSCL =============================== #
    parser.add_argument('--train_gail_lambda', '-tgl', action='store_true')
    parser.add_argument('--n_iters', '-ni', type=int, default=10000)
    # parser.add_argument('--warmup_timesteps', '-wt', type=lambda x: int(float(x)), default=None)
    # parser.add_argument('--forward_timesteps', '-ft', type=lambda x: int(float(x)), default=2e4)
    parser.add_argument('--backward_iters', '-bi', type=int, default=20)
    parser.add_argument('--backward_iters_extra', '-bie', type=int, default=1)
    parser.add_argument('--rollouts_per_demonstration', '-rpd', type=int, default=1)
    # note that with more backward_iters and backward_iters_extra it may already break in the middle because of the kl divergence.
    # parser.add_argument('--forward_iters', '-fi', type=int, default=1)  # default: 1, Baiyu: 60
    parser.add_argument('--no_importance_sampling', '-nis', type=bool, default=True)
    parser.add_argument('--per_step_importance_sampling', '-psis', action='store_true')
    # parser.add_argument('--reset_policy', '-rp', action='store_true')   # it creates new policies from scratch.
    parser.add_argument('--train_with_memory', '-twm', action='store_true')
    parser.add_argument('--policy_excerpts_weight', '-pew', type=float, default=1.)
    # ====================== constraint net ========================= #
    parser.add_argument('--cn_layers', '-cl', type=int, default=[4], nargs='*')
    parser.add_argument('--anneal_clr_by_factor', '-aclr', type=float, default=1.0)
    parser.add_argument('--cn_learning_rate', '-clr', type=float, default=0.03)
    # smaller learning rate is better to prevent breaking because of the KL divergence.
    parser.add_argument('--cn_reg_coeff', '-crc', type=float, default=0.0)
    parser.add_argument('--cn_batch_size', '-cbs', type=int, default=None)
    parser.add_argument('--cn_obs_select_dim', '-cosd', type=int, default=[0, 1], nargs='+')    # first two columns are the x and y positions
    parser.add_argument('--cn_acs_select_dim', '-casd', type=int, default=[-1], nargs='+')  # actions are not used in the constraint net
    parser.add_argument('--cn_plot_every', '-cpe', type=int, default=3)
    parser.add_argument('--cn_normalize', '-cn', action='store_true')
    parser.add_argument('--cn_target_kl_old_new', '-ctkon', type=float, default=10)
    parser.add_argument('--cn_target_kl_new_old', '-ctkno', type=float, default=100)
    parser.add_argument('--cn_eps', '-ce', type=float, default=1e-5)
    parser.add_argument('--cn_loss_type', '-clt', type=str, default='bce')  # options: ml, bce, sbce
    parser.add_argument('--cn_training_mode', '-ctm', type=str, default='pucl')  # options: pucl, mil
    parser.add_argument('--cn_mil_Tn', '-cmtn', type=float, default=0.2)
    parser.add_argument('--cn_mil_pooling', '-cmp', type=str, default='lse')  # options: lse, mean, min, noiseor
    parser.add_argument('--cn_initial_feasible_bias', '-cifb', type=float, default=None)
    parser.add_argument('--cn_manual_threshold', '-cmt', type=float, default=0.5)
    # ====================== PU learning ========================= #
    parser.add_argument('--refine_iterations', '-ri', type=int, default=1)
    parser.add_argument('--rn_decision_method', '-rdm', type=str, default='kNN')
    parser.add_argument('--kNN_k', '-kNNk', type=int, default=1)
    parser.add_argument('--kNN_thresh', '-kNNt', type=float, default=0.15)
    parser.add_argument('--kNN_normalize', '-kNNn', action='store_true')
    parser.add_argument('--kNN_metric', '-kNNm', type=str, default='euclidean') # choose from 'euclidean', 'manhattan', 'chebyshev', 'weighted_euclidean'
    parser.add_argument('--CPU_n_gmm_k', '-CPUngk', type=int, default=30)
    parser.add_argument('--CPU_ratio_thresh', '-CPUrt', type=float, default=0.8)
    parser.add_argument('--GPU_n_gmm', '-GPUng', type=int, default=30)
    parser.add_argument('--GPU_likelihood_thresh', '-GPUlt', type=float, default=-5)
    parser.add_argument('--WB_label_frequency', '-WBlf', type=float, default=1)
    parser.add_argument('--add_rn_each_traj', '-aret', action='store_true')
    # ======================== expert data ========================== #
    parser.add_argument('--expert_path', '-ep', type=str, default='icrl/expert_data/PointCircle')
    parser.add_argument('--expert_rollouts', '-er', type=int, default=20)   # max
    # ========================== various =========================== #
    parser.add_argument('--action_clip_value', '-acv', type=float, default=0.25)
    parser.add_argument('--position_limit', '-posl', type=float, default=13.0)
    parser.add_argument('--use_true_constraint', '-utc', action='store_true')
    parser.add_argument('--use_wandb', '-uwb', type=bool, default=True)
    parser.add_argument('--select_policy_excerpts', '-spe', action='store_true')
    parser.add_argument('--policy_episode_length', '-pel', type=int, default=250)
    parser.add_argument('--expert_episode_length', '-eel', type=int, default=150)
    parser.add_argument('--evaluation_episode_num', '-een', type=int, default=30)
    parser.add_argument('--show_figures', '-sf', action='store_true')
    # =============================================================== #

    default_config, mod_name = {}, ''
    # overwriting config file with parameters supplied through parser
    # order of priority: supplied through command line > specified in config file > default values in parser
    config = utils.merge_configs(default_config, parser, sys.argv[1:])
    # config = vars(argparse.Namespace(**config))

    if config['debug_mode']:  # this is for a fast debugging, use python train_icrl.py -dm to enable the debug model
        # config['device'] = 'cpu'
        config['verbose'] = 2  # the verbosity level: 0 no output, 1 info, 2 debug
        config['num_threads'] = 1
        config['backward_iters_extra'] = 1
        config['n_eval_episodes'] = 10
        config['save_every'] = 1
        config['sample_rollouts'] = 10
        config['sample_data_num'] = 500
        config['store_sample_num'] = 1000
        config['cn_batch_size'] = 3
        config['backward_iters'] = 2
        config['warmup_timesteps'] = 0
        config['n_iters'] = 5

    if config['cn_training_mode'] not in ('pucl', 'mil'):
        raise ValueError("cn_training_mode must be either 'pucl' or 'mil'.")

    if config['cn_training_mode'] == 'mil' and config['cn_initial_feasible_bias'] is None:
        config['cn_initial_feasible_bias'] = 1

    # generating the name by concatenating arguments with non-default values
    # default values are either the one specified in config file or in parser (if both
    # are present then the one in config file is prioritized)
    config['name'] = utils.get_name(parser, default_config, config, mod_name)

    if config['use_wandb']:
        import os
        # os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=config['project'], name=config['name'], config=config, dir='./icrl', group=config['group'])
        wandb.config.save_dir = wandb.run.dir
        config = wandb.config
        print(utils.colorize('configured folder %s for saving' % config.save_dir, color='green', bold=True))
        print(utils.colorize('name: %s' % config.name, color='green', bold=True))
        utils.save_dict_as_json(config.as_dict(), config.save_dir, 'config')
    else:
        save_dir = './icrl/wandb/temporary/'
        utils.del_and_make(save_dir)
        config['save_dir'] = save_dir
        config = utils.Dict2Class(config)

    # calling the icrl algorithm

    dscl(config)
    end = time.time()
    print(utils.colorize('total elapsed time: %05.2f hours' % ((end - start) / 3600),
                         color='green', bold=True))


if __name__=='__main__':
    main()
    # fi, reward remove, lambda, lr
