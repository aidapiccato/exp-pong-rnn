"""Environment class.
"""

import io
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


GRID_DIM = 10

class ExpPongEnv(gym.Env):
    """Environment for Exp-Pong."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ExpPongEnv, self).__init__()
        # Actions are either stay, move left, move right
        self.action_space = spaces.Discrete(n=3, start=-1)
        # Observations are a discretized version of the screen, where entries with -1 are hidden, ones with 0 
        # are visible but empty, and ones with 1 contain a ball
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(GRID_DIM, GRID_DIM), dtype=np.int)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, GRID_DIM), dtype=np.int)
        self.max_steps = 50

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1
        obs = self._next_observation()
        reward = self._get_reward()
        done = False
        if self.current_step > self.max_steps:
            done = True
        return dict(obs=obs, reward=reward, done=done, info={}, image=self._get_image())
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.agent_pos = np.int8(GRID_DIM/2)
        self.current_step = 0
        return dict(obs=self._next_observation(), reward=0, done=False, info={}, image=self._get_image())

    def _take_action(self, action):
        self.agent_pos += action 
        self.agent_pos = np.clip(self.agent_pos, a_min=0, a_max=GRID_DIM-1)

    def _get_reward(self):
        return self.agent_pos / GRID_DIM

    def _next_observation(self):
        obs = np.full(shape=(1, GRID_DIM), fill_value=-1)
        obs[:, self.agent_pos] = 0
        return obs

    def render(self, mode='human', close=False):
        print(f'Agent position: {self.agent_pos}')

    def _get_image(self):

        # return grid
        # plt.close('all')
        # fig = plt.figure()
        # plt.plot([1, 2])
        # plt.title("test")
        # buf = io.BytesIO()
        # plt.savefig(buf, format='jpeg')
        # buf.seek(0)
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()        
        grid = np.zeros(shape=(1, GRID_DIM))
        grid[:, self.agent_pos] = 1
        ax.imshow(grid)
        ax.axis('off')
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas.draw()       
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return image