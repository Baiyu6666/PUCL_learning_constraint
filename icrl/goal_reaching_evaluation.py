"""Episode-level evaluation for goal-conditioned Cartesian DS tasks."""

import numpy as np

from stable_baselines3.common.vec_env import VecEnv, VecNormalize


def evaluate_goal_reaching_policy(model, env, cost_function, n_episodes):
    """Evaluate one vector environment, retaining terminal state before auto-reset.

    Returns per-episode returns (not mean step reward), reach/safety rates and
    task records. The caller configures the task list and resets its cursor.
    Pass cost_function=None when true geometry is unknown; safety fields are null.
    """
    if not isinstance(env, VecEnv) or env.num_envs != 1:
        raise ValueError('goal-reaching evaluation requires a single vector environment')
    if isinstance(env, VecNormalize) and env.norm_obs:
        raise ValueError('goal-reaching evaluation expects physical, unnormalized observations')
    if n_episodes <= 0:
        raise ValueError('n_episodes must be positive')
    obs = env.reset()
    records = []
    has_true_cost = cost_function is not None
    for _ in range(n_episodes):
        initial = env.get_original_obs().copy() if isinstance(env, VecNormalize) else obs.copy()
        initial_cost = (np.asarray(cost_function(initial, np.zeros((1, env.action_space.shape[0])))).item()
                        if has_true_cost else None)
        safe = initial_cost == 0 if has_true_cost else None
        total_reward, total_cost, steps = 0.0, 0.0, 0
        state = None
        while True:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = bool(np.asarray(dones).item())
            info = infos[0]
            if done:
                if 'terminal_observation' not in info:
                    raise ValueError('vector environment did not preserve terminal_observation')
                next_original = np.asarray(info['terminal_observation'])[None, :]
            else:
                next_original = env.get_original_obs() if isinstance(env, VecNormalize) else obs
            cost = float(np.asarray(cost_function(next_original, action)).item()) if has_true_cost else 0.0
            if has_true_cost:
                safe = safe and cost == 0
            total_reward += float(np.asarray(reward).item())
            total_cost += cost
            steps += 1
            if done:
                reached = bool(info['goal_reached'])
                records.append({
                    'start': initial[0, :3].tolist(),
                    'goal': (initial[0, :3] + initial[0, 3:6]).tolist(),
                    'terminal_position': next_original[0, :3].tolist(),
                    'return': total_reward,
                    'steps': steps,
                    'violating_steps': total_cost if has_true_cost else None,
                    'safe': bool(safe) if has_true_cost else None,
                    'goal_reached': reached,
                    'path_length': float(info['path_length']),
                })
                break
        # obs already belongs to the next task: VecEnv reset it automatically.
    returns = [record['return'] for record in records]
    feasible_returns = [record['return'] for record in records if record['safe']]
    return {
        'average_reward': float(np.mean(returns)),
        'feasible_reward': (float(np.mean(feasible_returns)) if feasible_returns else np.nan) if has_true_cost else None,
        'violation_steps_portion': sum(r['violating_steps'] for r in records) / sum(r['steps'] for r in records) if has_true_cost else None,
        'safe_portion': float(np.mean([r['safe'] for r in records])) if has_true_cost else None,
        'goal_reached_portion': float(np.mean([r['goal_reached'] for r in records])),
        'safe_goal_reached_portion': float(np.mean([r['safe'] and r['goal_reached'] for r in records])) if has_true_cost else None,
        'average_episode_length': float(np.mean([r['steps'] for r in records])),
        'average_path_length': float(np.mean([r['path_length'] for r in records])),
        'episodes': records,
    }
