"""Environment class.
"""

import gym
import numpy as np
from gym import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.visualization import image_reshaping



class OccPongEnv(gym.Env):
    """Environment for occluded pong."""
    metadata = {'render.modes': ['human']}

    def __init__(self, p_prey, max_t, grid_height=10, grid_width=10, agent_gain=1, prey_gain=1, window_width=1):
        super(OccPongEnv, self).__init__()
        self._grid_width = grid_width
        self._grid_height = grid_height
        self.action_space = spaces.Discrete(n=3, start=-1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, self._grid_width), dtype=np.float32)
        self._p_prey = p_prey
        self._max_t = max_t
        self._agent_gain = agent_gain
        self._prey_gain = prey_gain
        self._window_width = window_width

    def reset(self): 
        # Reset the state of the environment to an initial state
        self._current_step = 0
        self._agent_pos = np.float32(self._grid_width/2)
        trajectory_len = int(self._grid_height/self._prey_gain)
        self._target_t = np.where(np.random.binomial(1, p=self._p_prey, size=self._max_t - trajectory_len))[0] + trajectory_len
        self._n_prey = len(self._target_t)
        self._target_x = [np.random.randint(low=0, high=self._grid_width) for _ in range(self._n_prey)]
        self._start_target_t = self._target_t - trajectory_len
        self._visible = [False for _ in range(self._n_prey)]
        self._input = self._generate_input()
        return dict(obs=self._next_observation(), reward=0, done=False)

    def _generate_input(self):
        input = np.zeros((self._max_t, self._grid_width))
        for start_target_t, target_t, target_x in zip(self._start_target_t, self._target_t, self._target_x):
            input[start_target_t:target_t, target_x] += np.linspace(0, 1, 10)
        return input

    def _next_observation(self):
        one_hot = np.zeros((1, self._grid_width))
        one_hot[:, int(self._agent_pos)] = 1
        input = self._input[self._current_step]

        for target_x, visible in zip(self._target_x, self._visible):
            if not visible:
                input[target_x] = 0                                

        return np.concatenate((input.reshape(1, -1), one_hot), axis=1).squeeze()

    def step(self, action):
        self._current_step += 1
        self._take_action(action)
        obs = self._next_observation()
        reward = self._get_reward()
        done = self._is_done()
        return dict(obs=obs, reward=reward, done=done)    

    def _is_done(self):
        return self._current_step == self._max_t - 1

    def _take_action(self, action):  
        action = action + self.action_space.start

        self._agent_pos += np.array(action).squeeze() * self._agent_gain

        self._agent_pos = np.clip(self._agent_pos, a_min=0, a_max=self._grid_width-1)

        for prey, target_x, start_target_t, target_t in zip(range(self._n_prey), self._target_x, self._start_target_t, self._target_t):
            if start_target_t <= self._current_step <= target_t:
                if target_x - self._window_width <= self._agent_pos <= target_x + self._window_width :
                    self._visible[prey] = True

    def _get_reward(self):
        reward = 0
        for target_x, target_t in zip(self._target_x, self._target_t):
            if self._current_step == target_t:
                reward += target_x == self._agent_pos
        return reward


    # def generate_episode_figure(self, agent, max_steps, buffer_height=3):
    #     num_steps = 0
    #     agent_pos = []
    #     reward = []
    #     input = []
    #     while num_steps == 0:
    #         timestep = self.reset()
    #         agent_pos = []
    #         reward = []
    #         input = []
    #         while not timestep['done'] and num_steps < max_steps:
    #             num_steps += 1
    #             action = agent.step(timestep, self, test=True)            
    #             timestep = self.step(action)
    #             agent_pos.append(self.agent_pos)
    #             reward.append(timestep['reward'])
    #             input.append(timestep['obs'])
    #         agent_pos = np.array(agent_pos).squeeze()
    #         reward = np.array(reward).squeeze()
    #         image = self._generate_episode_figure(agent_pos, self.target_x, self.target_t, reward, input)
    #     self.reset()
    #     images_dict = {'validation': image_reshaping(image, buffer_height)}    
    #     return images_dict

    # def _generate_episode_figure(self, agent_pos, target_x, target_t, reward, input):
    #     fig = Figure(figsize=(8, 4))
    #     canvas = FigureCanvas(fig)

    #     ax = fig.add_subplot(121)
    #     ax_right = ax.twinx()
    #     ax.plot(agent_pos)
    #     ax.scatter(target_t - 1, target_x, c='green')
    #     ax.set_ylim(-1, GRID_DIM)
    #     ax.set_ylabel('x position')
    #     ax.set_xlabel('time')
    #     ax_right.plot(reward, 'r')
    #     ax_right.set_ylim(-1, 1)
    #     ax_right.set_ylabel('reward')
    #     ax_right.yaxis.label.set_color('red')

    #     ax = fig.add_subplot(122)
    #     ax.imshow(input)
    #     ax.set_xlabel('x position')
    #     ax.set_ylabel('time')

    #     fig.tight_layout()
    #     width, height = fig.get_size_inches() * fig.get_dpi()
    #     canvas.draw()
    #     image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    #     return [image]

    def render(self, mode='human', close=False):
        print(f'Agent position: {self._agent_pos}')

    def generate_test_figure(self, agent, max_steps, buffer_height=3):
        return {}

    @property
    def data_keys(self):
        return ('obs', 'reward', 'done')
    def _get_image(self):
        return None
