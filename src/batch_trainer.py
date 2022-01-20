"""Trainer class."""

import os
from torch.utils import tensorboard
import torch
import logging
from envs.batch_env import BatchEnv
import numpy as np

class BatchTrainer:
    """Trainer class."""

    def __init__(self,
                 agent, 
                 batch_size,
                 env_class,
                 env_kwargs,
                 iterations,
                 scalar_eval_every,
                 train_every,
                 snapshot_every,
                 image_eval_every):
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
        self._env_class = env_class
        self._env_kwargs = env_kwargs
        self._batch_size = batch_size
        self._iterations = iterations
        self._train_every = train_every
        self._scalar_eval_every = scalar_eval_every
        self._snapshot_every = snapshot_every
        self._image_eval_every = image_eval_every


    def __call__(self, log_dir):
        """Run a training loop.
        
        Args:
            log_dir: String. Directory in which to log tensorboard logs for
                scalars and images.
        """
        env_batch = BatchEnv(batch_size=self._batch_size, env_class=self._env_class, **self._env_kwargs)

        timestep = env_batch.reset()

        agg_reward = 0.
        agg_loss = 0.

        snapshot_dir = os.path.join(log_dir, 'snapshots')
        os.makedirs(snapshot_dir)
        summary_dir = os.path.join(log_dir, 'tensorboard')
        summary_writer = tensorboard.SummaryWriter(log_dir=summary_dir)

        for step in range(self._iterations):
            action_batch = self._agent.step(timestep, env_batch)
            new_timestep = env_batch.step(action_batch)
            if step != 0 and step % self._train_every == 0:
                loss = self._agent.train_step(prev_timestep=timestep, action=action_batch, timestep=new_timestep)
                agg_loss += loss
            timestep = new_timestep.copy()

            # Update agg_reward
            if timestep['reward'] is not None:
                agg_reward += np.mean(timestep['reward'].numpy())

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

            # Log images if necessary (and only if an episode has been completed)
            if step % self._image_eval_every == 0 and torch.all(timestep['done']):
                logging.debug('Generating figure')
                # Train figure 
                episode_image_figure = env_batch.generate_episode_figure(agent=self._agent, max_steps=100)
                for k, v in episode_image_figure.items():
                    if 'z' in k:
                        summary_writer.add_image(k + ' train_figure', v, global_step=step)
                    else:
                        summary_writer.add_image(
                            k + ' train_figure', v, global_step=step, dataformats='HWC')

            # Reset environment if end of episode is reached
            if torch.all(timestep['done']): 
                logging.debug('Episode done')
                timestep = env_batch.reset()

            # Save snapshot
            if step % self._snapshot_every == 0:
                logging.info('Saving snapshot.')
                snapshot_filename = os.path.join(snapshot_dir, str(step))
                torch.save(self._agent.state_dict(), snapshot_filename)