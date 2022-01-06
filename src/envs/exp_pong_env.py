"""Environment class.
"""

import gym
import numpy as np
from gym import spaces

GRID_DIM = 5

class ExpPongEnv(gym.Env):
    """Environment for Exp-Pong."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ExpPongEnv, self).__init__()
        # Actions are either stay, move left, move right
        self.action_space = spaces.Discrete(n=3, start=-1)
        # Observations are a discretized version of the screen, where entries with -1 are hidden, ones with 0 
        # are visible but empty, and ones with 1 contain a ball
        self.observation_space = spaces.Box(low=-1, high=1, shape=(GRID_DIM, GRID_DIM), dtype=np.int)

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1
        obs = self._next_observation()
        reward = self._get_reward()
        done = False
        return dict(obs=obs, reward=reward, done=done, info={})
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.agent_pos = int(GRID_DIM/2)
        self.current_step = 0
        return self._next_observation()

    def _take_action(self, action):
        self.agent_pos += action 
        self.agent_pos = np.clip(self.agent_pos, a_min=0, a_max=GRID_DIM-1)

    def _get_reward(self):
        return self.agent_pos / GRID_DIM

    def _next_observation(self):
        obs = np.full(shape=(GRID_DIM, GRID_DIM), fill_value=-1)
        obs[:, self.agent_pos] = 0
        return obs

    def render(self, mode='human', close=False):
        print(f'Agent position: {self.agent_pos}')


