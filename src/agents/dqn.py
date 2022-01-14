"""DQN agent."""

import numpy as np
import torch

class DQN():

    def __init__(self, q_net, optim_config, num_actions, discount,
                 epsilon, batch_size, replay, grad_clip=None):
        self._q_net = q_net
        self._num_actions = num_actions
        self._replay = replay

        # Create optimizer
        self._optimizer = optim_config['optimizer'](
            self._q_net.parameters(), **optim_config['kwargs'])

        self._discount = discount
        self._epsilon = epsilon
        self._batch_size = batch_size

        self._loss = torch.nn.MSELoss()

        if grad_clip is not None and grad_clip <= 0:
            raise ValueError(f'grad_clip must be positive, but is {grad_clip}.')
        self._grad_clip = grad_clip

    def _to_one_hot(self, action):
        """Convert action to one-hot, because that gives faster learning."""
        batch_size = action.shape[0]
        one_hot_action = np.zeros((batch_size, self._num_actions))
        one_hot_action[np.arange(batch_size), action.astype(int)] = 1
        return one_hot_action

    def _eval_q_net(self, state, action):
        """Evaluate the Q net on a (state, action) pair.
        Args:
            state: Observation. Tensor of shape [batch_size, width_]
        Returns:
            Tensor of shape [batch_size].
        """
        # import pdb; pdb.set_trace()
        state = state.astype(np.float32) / 255.
        # if len(state.shape) < 3:
        #     state = np.expand_dims(state, axis=0)
        action = self._to_one_hot(action)
        # action = np.expand_dims(np.expand_dims(action, axis=1), axis=1)
        # action = np.tile(action, (1, state.shape[1], state.shape[2], 1))
        net_input = np.concatenate((state, action), axis=-1)
        # net_input = np.moveaxis(np.concatenate((state, action), axis=-1), -1, 1)
        net_input = torch.Tensor(net_input)
        return self._q_net(net_input)[:, 0]
    
    def _eval_all_actions(self, state):
        """Evaluate the Q net on all (state, action) pairs for a gives state.
        
        Returns:
            Numpy array of shape [batch_size, num_actions].
        """
        batch_ones = np.ones(state.shape[0])
        # q(s, a) for all a
        q_values = [
            self._eval_q_net(state, action * batch_ones).detach().numpy()
            for action in range(self._num_actions)
        ]
        return np.array(q_values)

    def step(self, timestep, env, test=False):
        del env
        if (np.random.rand() < self._epsilon) and not test:
            # Random action
            return np.random.randint(0, self._num_actions)
        else:
            # Best action according to q net
            obs = timestep['obs']
            q_values = self._eval_all_actions(np.array([obs]))
            best_action = np.argmax(q_values)
            return best_action

    def train_step(self, prev_timestep, action, timestep):
        # Write to replay
        self._replay.write(
            prev_timestep['obs'],
            action,
            timestep['reward'],
            timestep['obs'],
        )
        # import pdb; pdb.set_trace()
        # Sample from replay
        obs_prev, a, r, obs = self._replay.read(batch_size=self._batch_size)

        # Zero grad
        self._optimizer.zero_grad()

        # Compute loss
        q_values = self._eval_all_actions(obs)
        q = np.max(q_values, axis=0)
        target = torch.Tensor(r) + self._discount * q
        loss = self._loss(self._eval_q_net(obs_prev, a), target.detach())

        # Perform gradient update
        # AIDA: Performing gradient update on the q-value, using as target the true reward obtained when performing that action
        loss.backward()

        # Clip gradients if necessary
        if self._grad_clip is not None:
            for param in self._q_net.parameters():
                param.grad.data.clamp_(-self._grad_clip, self._grad_clip)
                
        self._optimizer.step()
        return loss

    def state_dict(self):
        return self._q_net.state_dict()

    def reset(self):
        pass