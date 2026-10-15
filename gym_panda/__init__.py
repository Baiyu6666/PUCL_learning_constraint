from gym.envs.registration import register

register(
    id='KukaCup1-v0',
    entry_point='gym_panda.envs:KukaCup1',
    max_episode_steps=200,
)

register(
    id='ReachObs-v0',
    entry_point='gym_panda.envs:ReachObs',
    max_episode_steps=400,
)

register(
    id='ReachVelObs-v0',
    entry_point='gym_panda.envs:ReachVelObs',
    max_episode_steps=100,
)

register(
    id='ReachVel-v0',
    entry_point='gym_panda.envs:ReachVel',
    max_episode_steps=100,
)

register(
    id='ReachRender-v0',
    entry_point='gym_panda.envs:ReachRender',
    max_episode_steps=400,
)

register(
    id='ReachConcaveObs-v0',
    entry_point='gym_panda.envs:ReachConcaveObs',
    max_episode_steps=300,
)

register(
    id='Reach2RegionObs-v0',
    entry_point='gym_panda.envs:Reach2RegionObs',
    max_episode_steps=300,
)

register(
    id='xarm-v0',
    entry_point='gym_panda.envs:XarmEnv',
)

register(
    id='kuka-v0',
    entry_point='gym_panda.envs:KukaEnv',
)
