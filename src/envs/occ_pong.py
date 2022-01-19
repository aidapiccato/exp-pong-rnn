"""Environment class.
"""

import gym
import numpy as np
from gym import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.visualization import image_reshaping

GRID_DIM = 10

class OccPongEnv(gym.Env):
    """Environment for occluded Pong."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(OccPongEnv, self).__init__()
        # Actions are either stay, move left, move right
        self.action_space = spaces.Discrete(n=3, start=-1)
        # Observations are an array of floats, where value of each one indicates amount of time         
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, GRID_DIM), dtype=np.float32)
        self.n_prey = 1
        self.min_target_t = 5
        self.min_int_t = 5
        self.max_int_t = 10
        self.max_t = 30
        self.target_x_distrib = spaces.Discrete(n=GRID_DIM)
        self.target_interval_t_distrib = spaces.Discrete(n=self.max_int_t - self.min_int_t, start=self.min_int_t)                
        self.agent_gain = 1
        self.prey_gain = GRID_DIM/10
        self.x_width = 0


    def reset(self): 
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.agent_pos = np.float32(GRID_DIM/2)
        self.target_t = [self.target_interval_t_distrib.sample() for _ in range(self.n_prey)]
        self.target_t = np.cumsum(self.target_t)
        self.target_t += self.min_target_t 
        self.start_target_t = self.target_t - int(GRID_DIM/self.prey_gain)
        self.target_x = [self.target_x_distrib.sample() for _ in range(self.n_prey)]
        self.seen = [False for _ in range(self.n_prey)]
        self.input = self._generate_input()
        return dict(obs=self._next_observation(), reward=0, done=False, info={}, image=self._get_image())

    def _generate_input(self):
        input = np.zeros((self.max_t, GRID_DIM))
        for start_target_t, target_t, target_x in zip(self.start_target_t, self.target_t, self.target_x):
            input[start_target_t:target_t, target_x] = np.linspace(0, 1, 10)
        return input

    def _next_observation(self):
        one_hot = np.zeros((1, GRID_DIM))
        one_hot[:, int(self.agent_pos)] = 1
        input = self.input[self.current_step]
        # Masking unseen balls
        for target_x, seen in zip(self.target_x, self.seen):
            if not seen:
                input[target_x] = 0                                
        return np.concatenate((input.reshape(1, -1), one_hot), axis=1).squeeze()

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        self._take_action(action)
        obs = self._next_observation()
        reward = self._get_reward()
        done = self._is_done()
        return dict(obs=obs, reward=reward, done=done, info={}, image=self._get_image())    

    def _is_done(self):
        return self.current_step > (np.amax(self.target_t) + 2) or self.current_step > self.max_t

    def _take_action(self, action):  
        action = action + self.action_space.start
        self.agent_pos += np.array(action).squeeze() * self.agent_gain
        self.agent_pos = np.clip(self.agent_pos, a_min=0, a_max=GRID_DIM-1)
        for prey, target_x, start_target_t, target_t, seen in zip(range(self.n_prey), self.target_x, self.start_target_t, self.target_t, self.seen):
            if start_target_t <= self.current_step <= target_t:
                if not seen:
                    if target_x - self.x_width <= self.agent_pos <= target_x + self.x_width :
                        self.seen[prey] = True

    def _get_reward(self):
        reward = 0
        for target_x, target_t in zip(self.target_x, self.target_t):
            if self.current_step == target_t:
                if target_x - self.x_width <= self.agent_pos <= target_x + self.x_width:
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
        fig = Figure(figsize=(8, 4))
        canvas = FigureCanvas(fig)

        ax = fig.add_subplot(121)
        ax_right = ax.twinx()
        ax.plot(agent_pos)
        ax.scatter(target_t - 1, target_x, c='green')
        ax.set_ylim(-1, GRID_DIM)
        ax.set_ylabel('x position')
        ax.set_xlabel('time')
        ax_right.plot(reward, 'r')
        ax_right.set_ylim(-1, 1)
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

    def render(self, mode='human', close=False):
        print(f'Agent position: {self.agent_pos}')

    def generate_test_figure(self, agent, max_steps, buffer_height=3):
        return {}

        
    def _get_image(self):
        return None
