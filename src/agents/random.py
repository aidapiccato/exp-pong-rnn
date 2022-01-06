"""Random agent."""

import numpy as np

class Random():

    def step(self, timestep, env, **kwargs):
        del timestep
        del kwargs

        return env.action_space.sample()

    def train_step(self, *args, **kwargs):
        del args
        del kwargs
        loss = 0.
        return loss

    def state_dict(self):
        return {} 
    
