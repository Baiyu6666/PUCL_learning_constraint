from functools import partial
import copy
import math

import custom_envs
import gym
import numpy as np


def get_true_cost_function(env_id):
    """Returns the cost function correpsonding to provided env)"""
    if env_id in ["HCWithPosTest-v0",
                  "WalkerWithPosTest-v0",
                  ]:
        return partial(wall_behind, -3)
        # return partial(wall_infront, 322)
    elif env_id == "AntWallTest-v0":
        return partial(wall_behind, -3)
    elif env_id == "SwimmerWithPosTest-v0":
        return partial(wall_behind, -0.5)
    elif env_id == "AntWallBrokenTest-v0":
        return partial(wall_behind, -3)
    elif env_id in ["PointNullRewardTest-v0", "PointCircleTest-v0", "AntCircleTest-v0"]:
        return partial(wall_behind_and_infront, -6, +6)
    elif env_id == "PointCircleTestBack-v0":
        return partial(wall_behind, -3)
    elif env_id in ["PointHalfCircleTest-v0"]:
        return partial(wall_behind, -6)
    elif env_id in ["PointObstacleTest-v0"]:
        return partial(rectangle_obstacle, -2, 5, -2, 2)
    elif env_id in ["PointObstacle2Test-v0"]:
        return partial(complex_obstacle,-2, 2.9, -1.9, 2, 3, 2, 2)
    elif env_id in ["CDD2B-v0", "CC2B-v0", "CDD3B-v0"]:
        dummy_env = gym.make(env_id)
        return partial(bridges, dummy_env)
    elif env_id == "CLGW-v0":
        return lap_grid_world
    elif env_id in ["AntTest-v0", 'HalfCheetahTest-v0', 'Walker2dTest-v0', 'SwimmerTest-v0']:
        return partial(torque_constraint, 0.5)
    elif env_id in ["PointDS-v0", 'PointDSTest-v0']:
        point_env = gym.make('PointDSTest-v0')
        return point_env.true_cost_function
    elif env_id in ["PointEllip-v0", 'PointEllipTest-v0']:
        point_env = gym.make('PointEllipTest-v0')
        return point_env.true_cost_function
    elif env_id in ["ReachObs-v0", "ReachVelObs-v0", "ReachConcaveObs-v0", "Reach2RegionObs-v0", "ReachVel-v0", "KukaCup1-v0"]:
        env = gym.make(env_id)
        return env.true_cost_function
    else:
        raise NotImplementedError(f"Cost function for {env_id} is not implemented yet.")
        # return null_cost

# ============================================================================
# General cost functions
# ============================================================================



def wall_behind(pos, obs, acs):
    return (obs[...,0] <= pos)

def wall_infront(pos, obs, acs):
    return (obs[...,0] >= pos)

def wall_behind_and_infront(pos_back, pos_front, obs, acs):
    return (obs[...,0] <= pos_back).astype(np.float32) + (obs[...,0] >= pos_front).astype(np.float32)

def rectangle_obstacle(pos_back, pos_front, pos_down, pos_up, obs, acs):
    return ((pos_front>=obs[...,0]) & (obs[...,0]>=pos_back) & (pos_up>=obs[...,1]) & (obs[...,1]>=pos_down)).astype(np.float32)

def complex_obstacle(pos_back, pos_front, pos_down, pos_up, cir_x, cir_y, r, obs, acs):
    return (((pos_front>=obs[...,0]) & (obs[...,0]>=pos_back) & (pos_up>=obs[...,1]) & (obs[...,1]>=pos_down)) | ((obs[:, 0]-cir_x)**2 + (obs[:,1]-cir_y)**2 <= r**2)).astype(np.float32)

def null_cost(x, *args):
    # Zero cost everywhere
    return np.zeros(x.shape[:1])

def torque_constraint(threshold, obs, acs):
    return np.any(np.abs(acs) > threshold, axis=-1)

# ============================================================================
# Specific cost functions
# ============================================================================

def bridges(env, obs, acs):
    from custom_envs.envs.utils import in_regions

    obs = copy.deepcopy(obs)
    acs = copy.deepcopy(acs)
    if len(obs.shape) > 2:
        batch_size, n_envs,_ = obs.shape
        cost_shape = (batch_size, n_envs)
        obs = np.reshape(obs, (batch_size*n_envs, -1))
        acs = np.reshape(acs, (batch_size*n_envs,-1))
    else:
        cost_shape = (obs.shape[0])

    cost = []
    for ob, ac in zip(obs, acs):
        # Need to unnormalize obs
        ob = unnormalize(env, ob)

        if 'action_map_dict' in env.__dict__:
            # For discrete classes
            ac = env.action_map_dict[int(ac)]
            next_ob = np.around(ob+ac, 6)
        else:
            # For continuous classes
            ac = np.clip(ac, a_min=env.action_space.low,
                         a_max=env.action_space.high)
            ori = ob[2] + ac[1]
            dx = math.cos(ori) * ac[0]
            dy = math.sin(ori) * ac[0]

            next_ob = ob.copy()
            next_ob[0] = np.clip(ob[0] + dx, -env.size, env.size)
            next_ob[1] = np.clip(ob[1] + dy, -env.size, env.size)

            ob = next_ob[:2]
            next_ob = next_ob[:2]

        if in_regions(ob, next_ob, env.constraint_regions):
            cost += [1]
        else:
            cost += [0]

    cost = np.reshape(np.array(cost), cost_shape)

    return cost

def lap_grid_world(obs, acs):
    cost = []
    for ac in acs:
        if ac == 1:
            cost += [1]
        else:
            cost += [0]
    return np.array(cost)


# ============================================================================
# Utils
# ============================================================================

def unnormalize(env, obs):
    if env.normalize:
        obs += 1
        obs *= (env.observation_space.high - env.observation_space.low)
        obs /= 2
        obs += env.observation_space.low
    return obs
