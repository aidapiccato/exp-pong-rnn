"""Environment class.
"""

import gym
import numpy as np
from gym import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.visualization import image_reshaping


GRID_DIM = 10

class ExpEnv(gym.Env):
    """Environment for Exp-Pong."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ExpEnv, self).__init__()
        # Actions are either stay, move left, move right
        self.action_space = spaces.Discrete(n=3, start=-1)
        # Observations are a discretized version of the screen, where entries with -1 are hidden, ones with 0 
        # are visible but empty, and ones with 1 contain a ball
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(GRID_DIM, GRID_DIM), dtype=np.int)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, GRID_DIM), dtype=np.int)
        self.max_steps = 20

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        self._take_action(action)
        obs = self._next_observation()
        reward = self._get_reward()
        done = False
        if self.current_step > self.max_steps:
            done = True
        return dict(obs=obs, reward=reward, done=done, info={}, image=None)
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.agent_pos = np.int8(GRID_DIM/2)
        self.last_visit = np.full(shape=(GRID_DIM), fill_value=-np.inf)
        self.prev_last_visit = np.copy(self.last_visit)
        self.last_visit[self.agent_pos] = self.current_step
        return dict(obs=self._next_observation(), reward=0, done=False, info={}, image=None)

    def _take_action(self, action):  
        action = action + self.action_space.start
        self.agent_pos += np.array(action).squeeze()
        self.agent_pos = np.clip(self.agent_pos, a_min=0, a_max=GRID_DIM-1)
        self.prev_last_visit = np.copy(self.last_visit)
        self.last_visit[self.agent_pos] = self.current_step

    def _get_reward(self):
        height = GRID_DIM
        time_since_last_visit = self.last_visit - self.current_step
        prev_time_since_last_visit = self.prev_last_visit - (self.current_step - 1)
        known = np.sum(np.clip(height + time_since_last_visit, a_min=0, a_max=height)) 
        prev_known = np.sum(np.clip(height + prev_time_since_last_visit, a_min=0, a_max=height)) 
        return np.clip(known - prev_known, a_min=0, a_max=np.inf)

    def _next_observation(self):
        one_hot = np.zeros((1, GRID_DIM))
        one_hot[:, int(self.agent_pos)] = 1
        one_hot = one_hot.squeeze()
        return one_hot
        # time_since_last_visit = self.last_visit - self.current_step
        # known = np.clip(GRID_DIM + time_since_last_visit, a_min=0, a_max=GRID_DIM)
        # return known.squeeze() 


    def render(self, mode='human', close=False):
        print(f'Agent position: {self.agent_pos}')

    def _get_image(self):
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()        
        grid = np.zeros(shape=(1, GRID_DIM))
        grid[:, self.agent_pos] = 1
        ax.imshow(grid)
        ax.axis('off')
        width, height = fig.get_size_inches() * fig.get_dpi()
        fig.tight_layout()
        canvas.draw()       
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return image

    def generate_episode_figure(self, agent, max_steps, buffer_height=3):
        num_steps = 0
        agent_pos = []
        reward = []
        input = []
        while num_steps == 0:
            timestep = self.reset()
            agent_pos = []
            reward = []
            input = []
            while not timestep['done'] and num_steps < max_steps:
                num_steps += 1
                action = agent.step(timestep, self, test=True)            
                timestep = self.step(action)
                agent_pos.append(self.agent_pos)
                reward.append(timestep['reward'])
                input.append(timestep['obs'])
            agent_pos = np.array(agent_pos).squeeze()
            reward = np.array(reward).squeeze()
            image = self._generate_episode_figure(agent_pos, reward, input)
        self.reset()
        images_dict = {'validation': image_reshaping(image, buffer_height)}    
        return images_dict

    def _generate_episode_figure(self, agent_pos, reward, input):
        fig = Figure(figsize=(8, 4))
        canvas = FigureCanvas(fig)

        ax = fig.add_subplot(121)
        ax_right = ax.twinx()
        ax.plot(agent_pos)
        ax.set_ylim(-1, GRID_DIM)
        ax.set_ylabel('x position')
        ax.set_xlabel('time')
        ax_right.plot(reward, 'r')
        ax_right.set_ylabel('reward')   
        ax_right.yaxis.label.set_color('red')
        
        ax = fig.add_subplot(122)
        ax.imshow(input)
        ax.set_xlabel('x position')
        ax.set_ylabel('time')

        fig.tight_layout()
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas.draw()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return [image]

    def generate_test_figure(self, agent, max_steps, buffer_height=3):
        return {}
