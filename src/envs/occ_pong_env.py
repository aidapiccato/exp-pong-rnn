"""Occluded pong environment.
"""

from this import d
import gym
import numpy as np
from gym import spaces


class OccPongEnv(gym.Env):
    """Environment for occluded pong."""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 p_prey,
                 n_steps,
                 grid_height=10,
                 grid_width=10,
                 agent_gain=1,
                 prey_gain=1,
                 window_width=1,
                 paddle_radius=1):
        super(OccPongEnv, self).__init__()
        self._grid_width = grid_width
        self._grid_height = grid_height
        self.action_space = spaces.Discrete(n=3, start=-1)
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(1, self._grid_width), dtype=np.float32)
        self._p_prey = p_prey
        self._n_steps = n_steps
        self._agent_gain = agent_gain
        self._prey_gain = prey_gain
        self._window_width = window_width
        self._paddle_radius = paddle_radius

    def reset(self):
        self._current_step = 0
        self._agent_pos = np.float32(self._grid_width/2)
        trajectory_len = int(self._grid_height/self._prey_gain)
        self._target_t = []
        while len(self._target_t) == 0:
            self._target_t = trajectory_len + \
                np.cumsum(np.random.geometric(
                    self._p_prey, size=self._n_steps))
            self._target_t = self._target_t[self._target_t < self._n_steps]
        self._n_prey = len(self._target_t)
        self._target_x = [np.random.randint(
            low=0, high=self._grid_width) for _ in range(self._n_prey)]
        self._start_target_t = self._target_t - trajectory_len
        self._input = self._generate_input()
        return dict(
            observation=self._next_observation(),
            reward=0,
            done=False,
            info=self._get_info()
        )

    def _generate_input(self):
        input = np.zeros((self._n_steps, self._grid_width))
        for start_target_t, target_t, target_x in zip(self._start_target_t, self._target_t, self._target_x):
            input[start_target_t:target_t,
                  target_x] += np.linspace(0, 1, int(self._grid_height/self._prey_gain))
        return input

    def _next_observation(self):
        one_hot = np.zeros((1, self._grid_width))
        one_hot[:, int(self._agent_pos)] = 1

        input = self._input[self._current_step]

        input_indices = np.arange(self._grid_width)
        input[input_indices > self._agent_pos + self._window_width] = 0
        input[input_indices < self._agent_pos - self._window_width] = 0

        return np.concatenate((input.reshape(1, -1), one_hot), axis=1).squeeze()

    def step(self, action):
        self._current_step += 1
        self._take_action(action)
        obs = self._next_observation()
        reward = self._get_reward()
        done = self._is_done()
        return dict(
            observation=obs,
            reward=reward,
            done=done,
            info=self._get_info()
        )

    def _get_info(self):
        info = dict(
            target_x=self._target_x,
            target_t=self._target_t,
            agent_pos=self._agent_pos
        )
        return info

    def _is_done(self):
        return self._current_step == self._n_steps - 1

    def _take_action(self, action):
        action = action + self.action_space.start

        self._agent_pos += np.array(action).squeeze() * self._agent_gain

        self._agent_pos = np.clip(
            self._agent_pos, a_min=0, a_max=self._grid_width-1)

    def _get_reward(self):
        reward = 0
        for target_x, target_t in zip(self._target_x, self._target_t):
            if self._current_step == target_t:
                caught = self._agent_pos - self._paddle_radius \
                    <= target_x <= self._agent_pos + self._paddle_radius
                reward += caught
        return reward

    @property
    def data_keys(self):
        return ('observation', 'reward', 'done', 'info')

    @property
    def n_steps(self):
        return self._n_steps

    def _get_image(self):
        return None
