"""Environment class.
"""

import gym
import numpy as np
from gym import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.visualization import image_reshaping

GRID_DIM = 5

class ExpPongEnv(gym.Env):
    """Environment for Pong."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ExpPongEnv, self).__init__()
        # Actions are either stay, move left, move right
        self.action_space = spaces.Discrete(n=3, start=-1)
        # Observations are an array of floats, where value of each one indicates amount of time         
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, GRID_DIM), dtype=np.float32)
        self.target_x_distrib = spaces.Discrete(n=GRID_DIM)
        self.max_steps = 50
        self.min_target_t = 5
        self.max_target_t = 20
        self.target_t_distrib = spaces.Discrete(n=self.max_target_t - self.min_target_t, start=self.min_target_t)
        self.agent_gain = 0.5
        self.x_width = 0

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.agent_pos = np.int8(GRID_DIM/2)
        self.target_x = self.target_x_distrib.sample()
        self.target_t = self.target_t_distrib.sample()
        self.input = self._generate_input()
        self.last_visit = np.full(shape=(GRID_DIM), fill_value=-np.inf)
        self.prev_last_visit = np.copy(self.last_visit)
        self.last_visit[self.agent_pos] = self.current_step
        return dict(obs=self._next_observation(), reward=0, done=False, info={}, image=self._get_image())

    def _generate_input(self):
        input = np.zeros((self.max_steps, GRID_DIM))
        step = 1/self.target_t # interval
        input[:, self.target_x] = step
        input = np.cumsum(input, axis=0)
        input[self.target_t:, :] = 0
        return input

    def _next_observation(self):
        time_since_last_visit = self.last_visit - self.current_step
        known = np.clip(GRID_DIM + time_since_last_visit, a_min=0, a_max=GRID_DIM)
        return np.concatenate((self.input[self.current_step, :].reshape(-1), known)).squeeze()
        # return known.squeeze()
        
    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        self._take_action(action)
        obs = self._next_observation()
        reward = self._get_reward()
        done = False
        if self.current_step > (self.target_t + 2):
            done = True
        return dict(obs=obs, reward=reward, done=done, info={}, image=self._get_image())    

    def _take_action(self, action):  
        action = action + self.action_space.start
        self.agent_pos += np.array(action).squeeze()
        self.agent_pos = np.clip(self.agent_pos, a_min=0, a_max=GRID_DIM-1)
        self.prev_last_visit = np.copy(self.last_visit)
        self.last_visit[self.agent_pos] = self.current_step

    def _get_intrinsic_reward(self):
        height = GRID_DIM
        time_since_last_visit = self.last_visit - self.current_step
        prev_time_since_last_visit = self.prev_last_visit - (self.current_step - 1)
        known = np.sum(np.clip(height + time_since_last_visit, a_min=0, a_max=height)) 
        prev_known = np.sum(np.clip(height + prev_time_since_last_visit, a_min=0, a_max=height)) 
        return np.clip(known - prev_known, a_min=0, a_max=np.inf)/GRID_DIM

    def _get_reward(self):
        reward = self._get_intrinsic_reward()
        if self.current_step == self.target_t:
            if self.target_x - self.x_width <= self.agent_pos <= self.target_x + self.x_width:
                reward += 1
        return reward

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
            agent_pos.append(self.agent_pos)
            reward.append(timestep['reward'])
            input.append(timestep['obs'])
            while not timestep['done'] and num_steps < max_steps:
                num_steps += 1
                action = agent.step(timestep, self, test=True)            
                timestep = self.step(action)

                agent_pos.append(self.agent_pos)
                reward.append(timestep['reward'])
                input.append(timestep['obs'])
            agent_pos = np.array(agent_pos).squeeze()
            reward = np.array(reward).squeeze()
            image = self._generate_episode_figure(agent_pos, self.target_x, self.target_t, reward, input)
        self.reset()
        images_dict = {'validation': image_reshaping(image, buffer_height)}    
        return images_dict

    def _generate_episode_figure(self, agent_pos, target_x, target_t, reward, input):
        fig = Figure()
        canvas = FigureCanvas(fig)

        ax = fig.add_subplot(121)
        ax_right = ax.twinx()
        ax.plot(agent_pos)
        ax.scatter(target_t, target_x, c='green')
        ax.set_ylim(-1, GRID_DIM)
        ax.set_ylabel('x')
        ax.set_xlabel('timestep')
        ax_right.plot(reward, 'r')
        ax_right.set_ylim(-1, 1)
        ax_right.set_ylabel('reward')

        ax = fig.add_subplot(122)
        ax.imshow(input)

        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas.draw()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return [image]

    def render(self, mode='human', close=False):
        print(f'Agent position: {self.agent_pos}')

    def _get_image(self):
        return None
