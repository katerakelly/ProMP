import numpy as np
from maml_zoo.utils.serializable import Serializable
from gym.spaces import Box
from rand_param_envs.gym.spaces import Box as OldBox


class RL2Env(Serializable):
    """
    Normalize action to in [-1, 1].
    Optionally normalize observations and actions
    """
    def __init__(
            self,
            env,
            normalize_obs=False,
            normalize_actions=False,
            obs_mean=None,
            obs_std=None,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        self.normalize_obs = normalize_obs
        if self.normalize_obs:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        self.normalize_actions = normalize_actions
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        d["_obs_mean"] = self._obs_mean
        d["_obs_std"] = self._obs_std
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_mean = d["_obs_mean"]
        self._obs_std = d["_obs_std"]

    def reset(self):
        obs = self._wrapped_env.reset()
        if self.normalize_obs:
            obs = self._apply_normalize_obs(obs)
        return np.concatenate([obs, np.zeros(self._wrapped_env.action_space.shape), [0], [0]])


    def step(self, action):
        scaled_action = action
        if self.normalize_actions:
            lb = self._wrapped_env.action_space.low
            ub = self._wrapped_env.action_space.high
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self.normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        next_obs = np.concatenate([next_obs, action, [reward], [done]])
        return next_obs, reward, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)

"""
Normalizes the environment class.

Args:
    EnvCls (gym.Env): class of the unnormalized gym environment
    env_args (dict or None): arguments of the environment
    scale_reward (float): scale of the reward
    normalize_obs (bool): whether normalize the observations or not
    normalize_reward (bool): whether normalize the reward or not
    obs_alpha (float): step size of the running mean and variance for the observations
    reward_alpha (float): step size of the running mean and variance for the observations

Returns:
    Normalized environment

"""


class _RL2Env(Serializable):
    """
    Normalizes the environment class.

    Args:
        Env (gym.Env): class of the unnormalized gym environment
        scale_reward (float): scale of the reward
        normalize_obs (bool): whether normalize the observations or not
        normalize_reward (bool): whether normalize the reward or not
        obs_alpha (float): step size of the running mean and variance for the observations
        reward_alpha (float): step size of the running mean and variance for the observations

    """
    def __init__(self,
                 env,
                 scale_reward=1.,
                 normalize_obs=False,
                 normalize_actions=True,
                 normalize_reward=False,
                 obs_alpha=0.001,
                 reward_alpha=0.001,
                 normalization_scale=10.,
                 ):
        Serializable.quick_init(self, locals())

        self._wrapped_env = env

    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        """
        orig_attr = self._wrapped_env.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr

    def reset(self):
        obs = self._wrapped_env.reset()
        return np.concatenate([obs, np.zeros(self._wrapped_env.action_space.shape), [0], [0]])

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)

    def step(self, action):
        wrapped_step = self._wrapped_env.step(action)
        next_obs, reward, done, info = wrapped_step
        next_obs = np.concatenate([next_obs, action, [reward], [done]])
        return next_obs, reward, done, info


rl2env = RL2Env
