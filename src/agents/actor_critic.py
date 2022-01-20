"""ActorCriticAgent """
import torch
from itertools import count


class ActorCritic():
    def __init__(self, actor_net, critic_net, state_net, num_actions, discount, optim_config):
        self._actor_net = actor_net
        self._critic_net = critic_net
        self._state_net = state_net
        self._num_actions = num_actions
        self._actor_optimizer = optim_config['actor_optimizer'](self._actor_net.parameters(), **optim_config['actor_kwargs'])
        self._critic_optimizer = optim_config['critic_optimizer'](self._critic_net.parameters(), **optim_config['critic_kwargs'])

        self._discount = discount

        self._discount = discount

    def reset():
        pass

    def _forward(self, obs):
        state = self._state_net(obs)
        dist = self._actor_net(state)
        action = dist.sample()
        value = self._critic_net(action, state)
        return dict(state=state, action=action, value=value))

    def train_episode(self, env):
        timestep = env.reset()
        values = []
        rewards = []
        
        for step in count():
            # get agent output

            # step environment

            # get reward and done from timestep

            # save reward and value to array

            # if done flag is true, break

            pass

        # get action and value for the current timestep 

        # compute returns

        # compute actor loss

        # compute critic loss

        # zero gradients for optimizers

        # backward pass


