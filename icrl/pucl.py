import argparse
import os
import sys
import time
import gym
import numpy as np
from stable_baselines3 import PPOLagrangian
from stable_baselines3.common import logger
from stable_baselines3.common.evaluation import evaluate_policy, evaluate_policy_with_cost
from stable_baselines3.common.logger import ERROR
from stable_baselines3.common.vec_env import sync_envs_normalization, VecNormalize
import icrl.utils as utils
from icrl.constraint_net import ConstraintNet, plot_constraints, evaluate_constraint_accuracy
from icrl.true_constraint_net import get_true_cost_function, null_cost
import gc

def pucl(config):
    if config.use_wandb:
        global wandb
        import wandb
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

    # setting specifications
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]
    action_low, action_high = None, None
    if isinstance(sampling_env.action_space, gym.spaces.Box):
        action_low, action_high = sampling_env.action_space.low, sampling_env.action_space.high

    # loading the expert data
    (expert_obs, expert_acs, expert_reward), expert_lengths, expert_mean_reward = utils.load_expert_data(
        config.expert_path, config.expert_rollouts, selection_seed=config.seed)

    # Set the starting point of the evaluation environment to be the same as the expert data for fair comparison
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
                                   weight_expert_loss=config.cn_weight_expert_loss,
                                   weight_nominal_loss=config.cn_weight_nominal_loss)
    if config.loading_constraint:
        constraint_net._load(f'icrl/wandb/{config.loading_constraint_dir}/files/models/icrl_{config.loading_constraint_ite}_itrs/cn.pt')
        print('Constraint net loaded from', f'icrl/wandb/{config.loading_constraint_dir}/files/models/icrl_{config.loading_constraint_ite}_itrs/cn.pt')

    # passing the cost function to the cost wrapper
    true_cost_function = get_true_cost_function(config.eval_env_id)
    if config.use_true_constraint:
        train_env.set_cost_function(true_cost_function)
        plot_cost_function = true_cost_function
    else:
        train_env.set_cost_function(constraint_net.cost_function)
        plot_cost_function = constraint_net.cost_function_non_binary

    cn_plot_dir = os.path.join(config.save_dir, 'constraint_net')
    video_dir = os.path.join(config.save_dir, 'video')
    utils.del_and_make(cn_plot_dir)

    # Plot True cost function and expert samples
    # plot_constraints(
    #         true_cost_function, config.cn_manual_threshold, config.position_limit, eval_env, config.eval_env_id, constraint_net.select_dim,
    #         obs_dim, acs_dim, os.path.join(config.save_dir, "true_constraint_net.png"), 'True Constraints and Expert Demonstration',
    #         expert_obs, expert_reward, expert_lengths
    # )

    # Initialize agent
    create_nominal_agent = lambda: PPOLagrangian(
            policy=config.policy_name,
            env=train_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            reward_gamma=config.reward_gamma,
            reward_gae_lambda=config.reward_gae_lambda,
            cost_gamma=config.cost_gamma,
            cost_gae_lambda=config.cost_gae_lambda,
            clip_range=config.clip_range,
            clip_range_reward_vf=config.clip_range_reward_vf,
            clip_range_cost_vf=config.clip_range_cost_vf,
            ent_coef=config.ent_coef,
            reward_vf_coef=config.reward_vf_coef,
            cost_vf_coef=config.cost_vf_coef,
            max_grad_norm=config.max_grad_norm,
            use_sde=config.use_sde,
            sde_sample_freq=config.sde_sample_freq,
            target_kl=config.target_kl,
            penalty_initial_value=config.penalty_initial_value,
            penalty_learning_rate=config.penalty_learning_rate,
            budget=config.budget,
            seed=config.seed,
            device=config.device,
            verbose=0,
            algo_type='pidlagrangian' if config.use_pid else 'lagrangian',
            pid_kwargs=dict(alpha=config.budget,
                                penalty_init=config.penalty_initial_value,
                                Kp=config.proportional_control_coeff,
                                Ki=config.integral_control_coeff,
                                Kd=config.derivative_control_coeff,
                                pid_delay=config.pid_delay,
                                delta_p_ema_alpha=config.proportional_cost_ema_alpha,
                                delta_d_ema_alpha=config.derivative_cost_ema_alpha,),
            policy_kwargs=dict(net_arch=utils.get_net_arch(config)),
            )
    nominal_agent = create_nominal_agent()

    # callbacks
    all_callbacks = []

    # warming-up or load existing policy
    timesteps = 0.
    if config.loading_policy:
        nominal_agent.set_parameters(f'icrl/wandb/{config.loading_policy_dir}/files/models/icrl_{config.loading_policy_ite}_itrs/nominal_agent', exact_match=True)
        train_env = VecNormalize.load(f'icrl/wandb/{config.loading_policy_dir}/files/models/icrl_{config.loading_policy_ite}_itrs/{config.loading_policy_ite}_train_env_stats.pkl', train_env.venv)
        print('Policy loaded from', f'icrl/wandb/{config.loading_policy_dir}/files/models/icrl_{config.loading_policy_ite}_itrs/nominal_agent')
    else:
        if config.warmup_timesteps is not None:
            print(utils.colorize('\nwarming up', color='green', bold=True))
            with utils.ProgressBarManager(config.warmup_timesteps) as callback:
                nominal_agent.learn(total_timesteps=config.warmup_timesteps,
                                    cost_function=null_cost, # do not incur any cost during warmp-up
                                    callback=callback)
                timesteps += nominal_agent.num_timesteps
            path = os.path.join(config.save_dir, f'models/icrl_0_itrs')
            utils.del_and_make(path)
            nominal_agent.save(os.path.join(path, f'nominal_agent'))
            constraint_net.save(os.path.join(path, f'cn.pt'))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(path, f'0_train_env_stats.pkl'))

    # training
    start_time = time.time()
    print(utils.colorize('\ntraining', color='green', bold=True), flush=True)
    best_true_reward, best_true_cost, best_forward_kl, best_reverse_kl = -np.inf, np.inf, np.inf, np.inf

    # Initialize the reliable infeasible data
    rn_obs = np.empty((0, expert_obs.shape[1]))

    for itr in range(config.n_iters):
        if config.reset_policy and itr != 0:
            print(utils.colorize('resetting the agent', color='green', bold=True), flush=True)
            nominal_agent = create_nominal_agent()

        # sampling nominal trajectories from the same starting point as that in expert data
        sync_envs_normalization(train_env, sampling_env)
        nominal_obs, full_nominal_obs, nominal_acs, nominal_rew, nominal_len = utils.sample_from_same_starting_points_as_demonstrations(
            nominal_agent, sampling_env, expert_obs, expert_lengths, config.rollouts_per_demonstration,
            deterministic=True, policy_excerpts=config.select_policy_excerpts, expert_reward=expert_reward,
            cost_function=constraint_net.cost_function, policy_excerpts_weight=config.policy_excerpts_weight)


        # Using the memory buffer
        if itr > 0 and config.use_memory:
            memory_obs = np.concatenate([nominal_obs, memory_obs])
            memory_acs = np.concatenate([nominal_acs, memory_acs])
            memory_len = np.concatenate([nominal_len, memory_len])
            print('Number of data in memory buffer', memory_obs.shape[0])
        else:
            memory_obs = nominal_obs
            memory_acs = nominal_acs
            memory_len = nominal_len

        # updating the constraint net (backward iterations)
        if config.cn_loss_type == 'pu':
            # For two-step PU learning method. Note that the method of identifying reliable infeasible data is also specified in the config, i.e., knn, cpu, gpu
            expert_distribution, rn_obs, backward_metrics = constraint_net.train_with_two_step_pu_learning(
                config.backward_iters, expert_obs, expert_acs, memory_obs, memory_acs, memory_len, config)
        else:
            # For MECL and BC method
            backward_metrics = constraint_net.train_MECL_BC(config.backward_iters, expert_obs, expert_acs, memory_obs,
                                                            memory_acs, memory_len, config.WB_label_frequency)

        # updating the policy (forward iterations)
        with utils.ProgressBarManager(config.forward_timesteps) as callback:
            nominal_agent.learn(total_timesteps=config.forward_timesteps,
                                cost_function='cost',   #? null_cost, 'cost', cost should come from the cost wrapper
                                callback=[callback]+all_callbacks)
            forward_metrics = logger.Logger.CURRENT.name_to_value
            timesteps += nominal_agent.num_timesteps

        # Save:
        # (1): Periodically save model, evaluate model
        if itr % config.eval_every == 0 or itr == config.n_iters-1:
            # Evaluate the policy in the true environment
            sync_envs_normalization(train_env, eval_env)
            eval_start = time.time()
            average_true_feasible_reward, average_true_reward, average_true_cost, true_safe_portion, average_episode_length = evaluate_policy_with_cost(
                nominal_agent, eval_env, true_cost_function, n_eval_episodes=config.evaluation_episode_num,
                deterministic=False)
            print('evaluation policy time: %0.2f secs' % (time.time() - eval_start))

            # Evaluate the accuracy of constraint net
            accuracy_metrics = evaluate_constraint_accuracy(config.train_env_id,
                                                            constraint_net.cost_function_evaluation, true_cost_function,
                                                            obs_dim, acs_dim)

        if itr % config.save_every == 0 or itr == config.n_iters-1:
            path = os.path.join(config.save_dir, f"models/icrl_{itr+1}_itrs")
            utils.del_and_make(path)
            nominal_agent.save(os.path.join(path, f'nominal_agent'))
            constraint_net.save(os.path.join(path, f'cn.pt'))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(path, f'{itr+1}_train_env_stats.pkl'))

        # updating the best metrics
        if average_true_reward > best_true_reward:
            best_true_reward = average_true_reward
        if average_true_cost < best_true_cost:
            best_true_cost = average_true_cost

        metrics = {
                "time(m)": (time.time()-start_time)/60,
                "iteration": itr,
                "timesteps": timesteps,
                # "true/forward_kl": forward_kl,
                "true/feasible_reward": average_true_feasible_reward,
                "true/average_reward": average_true_reward,
                "true/violation_episodes_portion": 1 - true_safe_portion,
                "true/violation_steps_portion": average_true_cost,
                "true/average_episode_length": average_episode_length,
                'backward/RN_trajectory_percent': nominal_len.size / expert_lengths.size,
            # "true/reverse_kl": reverse_kl,
                "best_true/best_reward": best_true_reward,
                "best_true/best_cost": best_true_cost,
                # "best_true/best_forward_kl": best_forward_kl,
                # "best_true/best_reverse_kl": best_reverse_kl
                }
        metrics.update({k.replace("train/", "forward/"): v for k, v in forward_metrics.items()})
        metrics.update(backward_metrics)
        metrics.update(accuracy_metrics)

        utils.append_local_metrics(config.save_dir, metrics)
        if config.use_wandb:
            wandb.log(metrics)

        # Plot constraint net periodically
        if (itr % config.cn_plot_every == 0) or (itr == config.n_iters-1):
            # if config.clip_obs is not None:
            #     obs_for_plot = np.clip(obs_for_plot, -config.clip_obs, config.clip_obs)
            save_name = os.path.join(cn_plot_dir, '%d.png' % (itr + 1))
            fig_title = f'Iteration {itr + 1} '
            plot_constraints(plot_cost_function, config.cn_manual_threshold, None, None, config.eval_env_id,
                             config.cn_obs_select_dim, obs_dim, acs_dim, save_name, fig_title, obs_expert=expert_obs,
                             rew_expert=None, len_expert=expert_lengths, obs_nominal=full_nominal_obs, obs_rn=rn_obs,
                             show=config.show_figures)

        # making video in mujoco
        # if ((itr % (config.make_video_every) == 0) or (itr == config.n_iters-1)) and itr != 0:
        #     num_episodes = 3 if itr < config.n_iters - 1 else 6
        #     create_video_start = time.time()
        #     sync_envs_normalization(train_env, eval_env)
        #     utils.eval_and_make_video(eval_env, nominal_agent, video_dir, '%d' % itr, num_episodes)
        #     print('Creating video time: %0.2f secs' % (time.time() - create_video_start))

    # Log the constraint plotting in wandb for easy access
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

    if config.sync_wandb and config.use_wandb:
        utils.sync_wandb(config.save_dir, 120)

    train_env.close()
    eval_env.close()
    sampling_env.close()

def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    # loading_policy_ite has been removed by Erfan.
    # ========================== setup ============================== #
    parser.add_argument('file_to_run', type=str)
    parser.add_argument('--config_file', '-cf', type=str, default=None)
    parser.add_argument('--project', '-p', type=str, default='ICRL-FE2')
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--group', '-g', type=str, default='test')
    parser.add_argument('--device', '-d', type=str, default='cpu')
    parser.add_argument('--verbose', '-v', type=int, default=0) # default: 2
    parser.add_argument('--sync_wandb', '-sw', action='store_true')
    parser.add_argument('--wandb_sweep', '-ws', type=bool, default=False)
    parser.add_argument('--debug_mode', '-dm', action='store_true')
    parser.add_argument('--non_training_mode', '-ntm', action='store_true')
    parser.add_argument('--loading_policy', '-lp', action='store_true')
    parser.add_argument('--loading_policy_dir', '-lpd', type=str)
    parser.add_argument('--loading_policy_ite', '-lpi', type=str, default='-1') # -1 represents the last iteration
    parser.add_argument('--loading_constraint', '-lc', action='store_true')
    parser.add_argument('--loading_constraint_dir', '-lcd', type=str, default=None)
    parser.add_argument('--loading_constraint_ite', '-lci', type=str, default='-1')
    # ======================== environments ========================= #
    parser.add_argument('--train_env_id', '-tei', type=str, default='PointCircle-v0')
    parser.add_argument('--eval_env_id', '-eei', type=str, default='PointCircleTest-v0')
    parser.add_argument('--dont_normalize_obs', '-dno', action='store_true')
    parser.add_argument('--dont_normalize_reward', '-dnr', action='store_true')
    parser.add_argument('--dont_normalize_cost', '-dnc', action='store_true')
    parser.add_argument('--seed', '-s', type=int, default=10)
    parser.add_argument('--clip_obs', '-co', type=int, default=np.inf)
    # ============================ cost ============================= #
    parser.add_argument('--cost_info_str', '-cis', type=str, default='cost')
    # ===================== policy networks ========================= #
    parser.add_argument('--policy_name', '-pn', type=str, default='TwoCriticsMlpPolicy')
    parser.add_argument('--shared_layers', '-sl', type=int, default=None, nargs='*')
    parser.add_argument('--policy_layers', '-pl', type=int, default=[64, 64], nargs='*')
    parser.add_argument('--reward_vf_layers', '-rvl', type=int, default=[64, 64], nargs='*')
    parser.add_argument('--cost_vf_layers', '-cvl', type=int, default=[64, 64], nargs='*')
    # ====================== policy training ======================== #
    parser.add_argument('--n_steps', '-ns', type=int, default=250)
    parser.add_argument('--batch_size', '-bs', type=int, default=128)
    parser.add_argument('--n_epochs', '-ne', type=int, default=10)
    parser.add_argument('--num_threads', '-nt', type=int, default=12)
    parser.add_argument('--save_every', '-se', type=float, default=5)
    parser.add_argument('--eval_every', '-ee', type=float, default=5)
    # ======================== policy mdp =========================== #
    parser.add_argument('--reward_gamma', '-rg', type=float, default=0.99)
    parser.add_argument('--reward_gae_lambda', '-rgl', type=float, default=0.95)
    parser.add_argument('--cost_gamma', '-cg', type=float, default=0.99)
    parser.add_argument('--cost_gae_lambda', '-cgl', type=float, default=0.95)
    # ====================== policy losses ========================== #
    parser.add_argument('--clip_range', '-cr', type=float, default=0.2)
    parser.add_argument('--clip_range_reward_vf', '-crv', type=float, default=None)
    parser.add_argument('--clip_range_cost_vf', '-ccv', type=float, default=None)
    parser.add_argument('--ent_coef', '-ec', type=float, default=0.0)
    parser.add_argument('--reward_vf_coef', '-rvc', type=float, default=0.5)
    parser.add_argument('--cost_vf_coef', '-cvc', type=float, default=0.5)
    parser.add_argument('--target_kl', '-tk', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', '-mgn', type=float, default=0.5)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    # ==================== policy lagrangian ======================== #
    # (a) General Arguments
    parser.add_argument('--use_pid', '-upid', action='store_true')
    parser.add_argument('--penalty_initial_value', '-piv', type=float, default=0.1)
    parser.add_argument('--budget', '-b', type=float, default=0.01)
    parser.add_argument('--update_penalty_after', '-upa', type=int, default=1)
    # (b) PID Lagrangian
    parser.add_argument('--proportional_control_coeff', '-kp', type=float, default=10)
    parser.add_argument('--derivative_control_coeff', '-kd', type=float, default=0)
    parser.add_argument('--integral_control_coeff', '-ki', type=float, default=0.5)
    parser.add_argument('--proportional_cost_ema_alpha', '-pema', type=float, default=0.5)
    parser.add_argument('--derivative_cost_ema_alpha', '-dema', type=float, default=0.5)
    parser.add_argument('--pid_delay', '-pidd', type=int, default=1)
    # (c) Traditional Lagrangian
    parser.add_argument('--penalty_learning_rate', '-plr', type=float, default=0.,
                        help='Sets Learning Rate of Dual Variables if use_pid is not true.')
    # ==================== policy exploration ======================= #
    parser.add_argument('--use_sde', '-us', action='store_true')
    parser.add_argument('--sde_sample_freq', '-ssf', type=int, default=-1)
    parser.add_argument('--use_curiosity_driven_exploration', '-ucde', action='store_true')
    # ========================== pucl =============================== #
    parser.add_argument('--train_gail_lambda', '-tgl', action='store_true')
    parser.add_argument('--n_iters', '-ni', type=int, default=50)
    parser.add_argument('--warmup_timesteps', '-wt', type=lambda x: int(float(x)), default=None)
    parser.add_argument('--forward_timesteps', '-ft', type=lambda x: int(float(x)), default=2e5)
    parser.add_argument('--backward_iters', '-bi', type=int, default=1000)
    parser.add_argument('--rollouts_per_demonstration', '-rpd', type=int, default=1)
    parser.add_argument('--select_policy_excerpts', '-spe', action='store_true')
    parser.add_argument('--policy_excerpts_weight', '-pew', type=float, default=1.)
    parser.add_argument('--no_importance_sampling', '-nis', action='store_true')
    parser.add_argument('--per_step_importance_sampling', '-psis', action='store_true')
    parser.add_argument('--reset_policy', '-rp', action='store_true')   # it creates new policies from scratch.
    # ====================== constraint net ========================= #
    parser.add_argument('--cn_layers', '-cl', type=int, default=[32, 32], nargs='*')
    parser.add_argument('--anneal_clr_by_factor', '-aclr', type=float, default=1.0)
    parser.add_argument('--cn_learning_rate', '-clr', type=float, default=0.003)
    # smaller learning rate is better to prevent breaking because of the KL divergence.
    parser.add_argument('--cn_reg_coeff', '-crc', type=float, default=0.0)   
    parser.add_argument('--cn_batch_size', '-cbs', type=int, default=None)
    parser.add_argument('--cn_obs_select_dim', '-cosd', type=int, default=[0, 1], nargs='+')    # For most envs, the first two columns are the x and y positions
    parser.add_argument('--cn_acs_select_dim', '-casd', type=int, default=[-1], nargs='+')  # -1 means no actions
    parser.add_argument('--cn_plot_every', '-cpe', type=int, default=3)
    parser.add_argument('--cn_normalize', '-cn', action='store_true')
    parser.add_argument('--cn_target_kl_old_new', '-ctkon', type=float, default=10)
    parser.add_argument('--cn_target_kl_new_old', '-ctkno', type=float, default=100)
    parser.add_argument('--cn_eps', '-ce', type=float, default=1e-5)
    parser.add_argument('--cn_loss_type', '-clt', type=str, default='pu')  # options: ml (MECL), bce (BCE), pu (two-step PU)
    parser.add_argument('--cn_manual_threshold', '-cmt', type=float, default=0.5)
    parser.add_argument('--cn_weight_expert_loss', '-cwelf', type=float, default=1.0)
    parser.add_argument('--cn_weight_nominal_loss', '-cwnl', type=float, default=1.0)
    # ======================== expert data ========================== #
    parser.add_argument('--expert_path', '-ep', type=str, default='icrl/expert_data/PointCircle')
    parser.add_argument('--expert_rollouts', '-er', type=int, default=999)
    # ====================== PU learning ========================= #
    parser.add_argument('--refine_iterations', '-ri', type=int, default=1)
    parser.add_argument('--rn_decision_method', '-rdm', type=str, default='kNN') #options: 'kNN', 'CPU', 'GPU'
    parser.add_argument('--kNN_k', '-kNNk', type=int, default=1)
    parser.add_argument('--kNN_thresh', '-kNNt', type=float, default=0.12)
    parser.add_argument('--kNN_normalize', '-kNNn', action='store_true')
    parser.add_argument('--kNN_metric', '-kNNm', type=str, default='euclidean') # options: 'euclidean', 'manhattan', 'chebyshev', 'weighted_euclidean'
    parser.add_argument('--CPU_n_gmm_k', '-CPUngk', type=int, default=20)
    parser.add_argument('--CPU_ratio_thresh', '-CPUrt', type=float, default=0.8)
    parser.add_argument('--GPU_n_gmm', '-GPUng', type=int, default=7)
    parser.add_argument('--GPU_likelihood_thresh', '-GPUlt', type=float, default=-6)
    parser.add_argument('--WB_label_frequency', '-WBlf', type=float, default=1)
    parser.add_argument('--add_rn_each_traj', '-aret', action='store_true', help='add at least one reliable infeasible data from each trajectory')
    # ========================== various =========================== #
    # parser.add_argument('--action_clip_value', '-acv', type=float, default=0.25)
    # parser.add_argument('--position_limit', '-posl', type=float, default=13.0)
    parser.add_argument('--use_true_constraint', '-utc', action='store_true')
    parser.add_argument('--use_wandb', '-uwb', action='store_true', default=False)
    parser.add_argument('--no_wandb', '-dwb', dest='use_wandb', action='store_false')
    parser.add_argument('--use_memory', '-um', action='store_true')
    parser.add_argument('--evaluation_episode_num', '-een', type=int, default=20)
    parser.add_argument('--make_video_every', '-mve', type=int, default=99)
    parser.add_argument('--show_figures', '-sf', action='store_true')
    # =============================================================== #

    default_config, mod_name = {}, ''
    # overwriting config file with parameters supplied through parser
    # order of priority: supplied through command line > specified in config file > default values in parser
    config = utils.merge_configs(default_config, parser, sys.argv[1:])
    # config = vars(argparse.Namespace(**config))

    if config['debug_mode']:  # this is for a fast debugging
        config['verbose'] = 2  # the verbosity level: 0 no output, 1 info, 2 debug
        config['num_threads'] = 1
        config['forward_timesteps'] = 500
        config['forward_iters'] = 1
        config['backward_iters_extra'] = 1
        config['n_steps'] = 100
        config['n_epochs'] = 2
        config['n_eval_episodes'] = 10
        config['save_every'] = 1
        config['sample_rollouts'] = 10
        config['sample_data_num'] = 500
        config['store_sample_num'] = 1000
        config['cn_batch_size'] = 3
        config['backward_iters'] = 2
        config['warmup_timesteps'] = 0
        config['n_iters'] = 5

    if config['non_training_mode']:  # In this mode, if the policy/constraint net is loaded, it won't be updated.
        config['n_iters'] = 10
        if config['loading_policy']:
            print('Warning - non_training mode is on and the policy is loaded, it wont be updated.')
            config['num_threads'] = 1
            config['forward_timesteps'] = 1
            config['forward_iters'] = 1
            config['n_steps'] = 100
            config['n_epochs'] = 2
            config['n_eval_episodes'] = 50
            config['warmup_timesteps'] = 0
            config['learning_rate'] = 0
        if config['loading_constraint']:
            print('Warning - non_training mode is on and the constraint is loaded, it wont be updated.')
            config['cn_learning_rate'] = 0
            config['backward_iters'] = 1
        # config['show_figures'] = True

    # generating the name by concatenating arguments with non-default values
    # default values are either the one specified in config file or in parser (if both
    # are present then the one in config file is prioritized)
    config['name'] = utils.get_name(parser, default_config, config, mod_name)

    if config['use_wandb']:
        global wandb
        import wandb
        import os
        # os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=config['project'], name=config['name'], config=config, dir='./icrl', group=config['group'])
        wandb.config.save_dir = wandb.run.dir
        config = wandb.config
        print(utils.colorize('configured folder %s for saving' % config.save_dir, color='green', bold=True))
        print(utils.colorize('name: %s' % config.name, color='green', bold=True))
        utils.save_dict_as_json(config.as_dict(), config.save_dir, 'config')
    else:
        config = utils.create_local_run(config)

    # calling the pucl algorithm
    pucl(config)
    end = time.time()
    print(utils.colorize('total elapsed time: %05.2f hours' % ((end - start) / 3600),
                         color='green', bold=True))
    gc.collect()


if __name__=='__main__':
    main()
