"""Random agent."""

import numpy as np

class Random():

    def step(self, timestep, env, **kwargs):
        del kwargs
        batch_size = timestep['obs'].size(0)
        return [env.action_space.sample() for _ in range(batch_size)]

    def train_step(self, *args, **kwargs):
        del args
        del kwargs
        loss = 0.
        return loss

    def state_dict(self):
        return {} 
    
