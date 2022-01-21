import gym
import numpy as np
import torch

class BatchEnv(gym.Env):
    def __init__(self, batch_size, env_class, **env_kwargs):
        self._batch_size = batch_size
        self._env_class = env_class
        self._env_kwargs = env_kwargs
        self._envs = [
            self._env_class(**self._env_kwargs) for _ in range(self._batch_size)
        ]

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def action_space(self):
         return self._envs[0].action_space

    @property 
    def n_steps(self):
        # TODO: Make prop. in occ_pong_env
        return self._envs[0]._n_steps

    @property
    def num_actions(self):
        return self.action_space.n

    def sample_action(self):
        return np.array([
            self.action_space.sample() for _ in range(self._batch_size)
        ])

    def reset(self):

        obs_batch = [env.reset() for env in self._envs]
        obs_batch = {k: np.stack([d[k] for d in obs_batch])
                      for k in self._envs[0].data_keys}
        return obs_batch


    def step(self, action_batch):
        obs_batch = [env.step(action) for (env, action) in zip(self._envs, action_batch)]
        obs_batch = {k: np.stack([d[k] for d in obs_batch])
                      for k in self._envs[0].data_keys}
        return obs_batch
 