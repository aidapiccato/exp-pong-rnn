"""Trainer class."""

import os
from torch.utils import tensorboard
import torch
import logging

class Trainer():
    """Trainer class."""

    def __init__(self,
                 agent,
                 env,
                 iterations,
                 scalar_eval_every,
                 snapshot_every):
        """Constructor.
        Args:
            agent: Instance of agent. Must have methods:
                * step(self, timestep, env, test=False) returning action.
                * train_step(self, prev_timestep, action, timestep) returning
                    loss.
            env: Instance of oog.environment.Environment. Environment used for
                training.
            iterations: Int. Number of training iterations.
            scalar_eval_every: Int. Periodicity of logging scalars.
        """
        self._agent = agent
        self._env = env
        self._iterations = iterations
        self._scalar_eval_every = scalar_eval_every
        self._snapshot_every = snapshot_every

    def __call__(self, log_dir):
        """Run a training loop.
        
        Args:
            log_dir: String. Directory in which to log tensorboard logs for
                scalars and images.
        """
        timestep = self._env.reset()

        agg_reward = 0.
        agg_loss = 0.

        snapshot_dir = os.path.join(log_dir, 'snapshots')
        os.makedirs(snapshot_dir)
        summary_dir = os.path.join(log_dir, 'tensorboard')
        summary_writer = tensorboard.SummaryWriter(log_dir=summary_dir)

        for step in range(self._iterations):

            # Take a step
            action = self._agent.step(timestep, self._env)
            new_timestep = self._env.step(action)
            if step != 0:
                loss = self._agent.train_step(timestep, action, new_timestep)
                agg_loss += loss
            timestep = new_timestep

            # Update agg_reward
            if timestep['reward'] is not None:
                agg_reward += timestep['reward']

            # Log scalars if necessary
            if step % self._scalar_eval_every == 0:
                # Training scalars
                scalars = {
                    'step': step,
                    'train_reward': agg_reward / self._scalar_eval_every,
                    'loss': agg_loss / self._scalar_eval_every,
                }

                # Log scalars
                logging.info(scalars)
                for k, v in scalars.items():
                    summary_writer.add_scalar(k, v, global_step=step)
                agg_reward = 0.
                agg_loss = 0.

