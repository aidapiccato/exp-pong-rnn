class BatchEnv:
    def __init__(self, batch_size, env_class, **env_kwargs):
        self._batch_size = batch_size
        self._env_class = env_class
        self._env_kwargs = env_kwargs
    
    def reset(self):
        self._envs = [self._env_class(self._env_kwargs) for _ in range(self._batch_size)]
        self._envs = [env.reset() for env in self._envs]

    def step(self, action_batch):
        obs_batch = [env.step(action) for (env, action) in zip(self._envs, action_batch)]
        return obs_batch
