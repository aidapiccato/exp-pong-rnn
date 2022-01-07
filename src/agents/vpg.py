"""Vanilla policy gradient (VPG) algorithm."""
import torch
from torch.distributions import Categorical
import numpy as np

class VPG():
    def __init__(self, policy_net, num_actions, optim_config, discount, epsilon, grad_clip, batch_size):
        """Constructor."""
        self._policy_net = policy_net
        self._num_actions = num_actions
        self._optimizer = optim_config['optimizer'](self._policy_net.parameters(), **optim_config['kwargs'])
        self._discount = discount
        self._epsilon = epsilon
        self._loss = torch.nn.MSELoss()
        self._batch_size = batch_size
        if grad_clip is not None and grad_clip <= 0:
            raise ValueError(f'grad_clip must be positive, but is {grad_clip}.')
        self._grad_clip = grad_clip
        self._action_batch = []
        self._state_batch = []
        self._reward_batch = []
        self._prev_action = 0


    def step(self, timestep, env, **kwargs):
        del env 

        if (np.random.rand() < self._epsilon):
            # Random action
            action = np.random.randint(0, self._num_actions)
        else:
            # Randomly sample action according to policy net
            obs = timestep['obs']
            probs = self._policy_net(torch.tensor(obs).unsqueeze(0).float())
            sampler = Categorical(probs)
            action = sampler.sample()
            action = action.detach().numpy()
        reward = timestep['reward']
        if timestep['done']:
            reward = 0
        self._action_batch.append(self._prev_action)
        self._state_batch.append(timestep['obs'])
        self._reward_batch.append(reward)
        self._prev_action = action
        return action



    def train_step(self, prev_timestep, action, timestep):
        """Training step.

        Args:
            prev_timestep ([type]): [description]
            action ([type]): [description]
            timestep ([type]): [description]
        """



        # Zero gradients
        self._optimizer.zero_grad()

        # Discount reward
        discounted_reward_batch = []
        running_sum = 0
        for reward in reversed(self._reward_batch):
            reward = np.float64(reward)
            if reward == 0:
                running_sum = 0
            else:
                running_sum = running_sum * self._discount + reward
                discounted_reward_batch.append(running_sum)
        discounted_reward_batch = np.asarray(list(reversed(discounted_reward_batch)))
        # Normalize reward
        reward_mean = np.mean(discounted_reward_batch)
        reward_std = np.std(discounted_reward_batch)       
        if reward_std < 1e-5:
            reward_std = 1.0
        discounted_reward_batch = (discounted_reward_batch - reward_mean) / reward_std
        agg_loss = 0.
        # Gradient descent
        for obs, action, reward in zip(self._state_batch, self._action_batch, discounted_reward_batch):
            probs = self._policy_net(torch.Tensor(obs))
            reward = torch.tensor(reward, requires_grad=False)
            sampler = Categorical(probs)
            loss = -sampler.log_prob(torch.tensor([action], requires_grad=False)) * reward
            agg_loss += loss.detach().numpy().squeeze()
            loss.backward()

            # Clip gradients if necessary
            if self._grad_clip is not None:
                for param in self._policy_net.parameters():
                    param.grad.data.clamp_(-self._grad_clip, self._grad_clip)
                    
        self._optimizer.step()
        self._action_batch = []
        self._state_batch = []
        self._reward_batch = []
        return agg_loss

    def state_dict(self):
        pass
