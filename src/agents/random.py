"""Random agent."""


class Random():

    def step(self, timestep, env, **kwargs):
        del kwargs
        batch_size = timestep['obs'].size(0)
        return [env.action_space.sample() for _ in range(batch_size)]

    def train_episode(self, *args, **kwargs):
        del args
        del kwargs
        loss = 0.
        reward = 0.
        return dict(loss=loss, reward=reward)

    def state_dict(self):
        return {} 
    
