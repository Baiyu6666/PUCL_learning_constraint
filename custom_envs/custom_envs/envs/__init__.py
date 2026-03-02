"""
Keep env package imports lazy.

Gym uses string entry points in ``custom_envs.__init__`` and imports the target
module only when ``gym.make(env_id)`` is called. Eager wildcard imports here
force MuJoCo-based modules to load even for unrelated environments.
"""
