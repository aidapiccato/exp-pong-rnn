"""QLearner agent class"""

import torch
import numpy as np


class QLearner(torch.nn.Module):
    def __init__(self,
                 env,
                 encoder,
                 rnn_core,
                 q_net,
                 discount,
                 epsilon,
                 optim_config):
        """Constructor.
        
            TODO:  Add docstring.
        """
        super(QLearner, self).__init__()
        self._env = env

        self._n_steps = self._env.n_steps

        self._num_actions = self._env.num_actions

        self.add_module('_encoder', encoder)
        self.add_module('_rnn_core', rnn_core)
        self.add_module('_q_net', q_net)

        self._discount = discount
        self._epsilon = epsilon
        
        params = (
            list(self._encoder.parameters()) +
            list(self._rnn_core.parameters()) +
            list(self._q_net.parameters())
        )

        self._optimizer = optim_config['optimizer'](
            params=params, **optim_config['kwargs'])

        discount_matrix = np.zeros((self._n_steps, self._n_steps))
        discount_coeffs = np.power(
            [1 - discount], np.arange(self._n_steps))
        for i in range(self._n_steps):
            discount_matrix[i, i:] = discount_coeffs[: self._n_steps - i]
        self._discount_matrix = torch.from_numpy(
            discount_matrix.astype(np.float32))

        self._loss_fn = torch.nn.MSELoss()

    def _eval_q_net(self, hidden, action):
        """Evaluate the Q net on a (hidden, action) pair.

        Args:
            hidden: Hidden state. Tensor of shape [batch_size, hidden_size].
            action: One-hot actions. Numpy array of shape [batch_size].

        Returns:
            Tensor of shape [batch_size].
        """
        action = torch.from_numpy(self._to_one_hot(action).astype(np.float32))
        net_input = torch.cat((hidden, action), axis=1)
        return self._q_net(net_input)[:, 0]

    def _to_one_hot(self, action):
        """Convert numpy action to one-hot."""
        batch_size = action.shape[0]
        one_hot_action = np.zeros((batch_size, self._num_actions))
        one_hot_action[np.arange(batch_size), action.astype(int)] = 1
        return one_hot_action

    def _eval_all_actions(self, hidden):
        """Evaluate the Q net on all (state, action) pairs for a gives state.
        
        Returns:
            Numpy array of shape [batch_size, num_actions].
        """
        batch_ones = np.ones(hidden.shape[0])
        q_values = np.array([
            self._eval_q_net(hidden, action * batch_ones).detach().numpy()
            for action in range(self._num_actions)
        ])
        return np.array(q_values)

    def _get_action(self, hidden, test=False):
        """Run the policy one step to get an action.
        
        Args:
            hidden: Hidden state. Tensor of shape [batch_size, hidden_size].
            test: Bool.

        Returns:
            actions: One-hot numpy array of actions. Shape
                [batch_size, num_actions].
        """

        if not test and np.random.rand() < self._epsilon:
            actions = self._env.sample_action()
        else:
            # Best action according to q net
            q_values = self._eval_all_actions(hidden)
            actions = np.argmax(q_values, axis=0)
        return actions

    def _init_hiddens(self):
        batch_size = self._env.batch_size
        hidden_size = self._rnn_core.out_features
        return torch.zeros([batch_size, hidden_size])

    def forward(self, test=False):
        """Run agent for an episode."""

        hiddens = [self._init_hiddens()]
        observations = []
        actions = []
        infos = []
        q_values = []

        timestep = self._env.reset()
        rewards = [timestep['reward']]
        for i in range(self._n_steps - 1):
            current_obs = torch.from_numpy(
                timestep['observation'].astype(np.float32))
            observations.append(current_obs)
            infos.append(timestep['info'])
            # Step RNN to get new hiddens
            encoded_obs = self._encoder(current_obs)
            new_hidden = self._rnn_core(
                torch.cat([encoded_obs, hiddens[-1]], dim=1))
            hiddens.append(new_hidden)

            # Step policy
            a = self._get_action(new_hidden, test=test)
            actions.append(a)

            # Compute q values
            q = self._eval_q_net(new_hidden, a)
            q_values.append(q)

            # Step environment on acctions and get reward
            timestep = self._env.step(a)
            rewards.append(timestep['reward']) 
        outputs = dict(
            rewards=rewards,
            hiddens=hiddens,
            observations=observations,
            actions=actions,
            q_values=q_values,
            infos=infos
        )

        return outputs


    def _loss(self, forward_outputs):
        rewards = torch.from_numpy(
            np.stack(forward_outputs['rewards'], axis=1).astype(np.float32))
        discounted_rewards = torch.matmul(rewards, self._discount_matrix)

        q_values = torch.stack(forward_outputs['q_values'], axis=1)
        loss = self._loss_fn(discounted_rewards[:, 1:], q_values)

        return loss

    def train(self):
        outputs = self.forward(test=False)
        loss = self._loss(outputs)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def scalars(self):
        
        outputs = self.forward(test=False)
        loss = self._loss(outputs)

        mean_reward = np.mean(outputs['rewards'])

        return {'loss': loss, 'mean_reward': mean_reward}









