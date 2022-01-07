"""Trainer class."""

import os
from torch.utils import tensorboard
import torch
import logging
from utils import visualization

class Trainer():
    """Trainer class."""

    def __init__(self,
                 agent,
                 env,
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
        self._env = env
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
            if step != 0 and step % self._train_every == 0:
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

            # Log images if necessary
            if step % self._image_eval_every == 0:
                # Train video
                episode_images_train = visualization.generate_episode_video(
                    self._env, self._agent, max_steps=100)
                # episode_images_train = list(episode_images_train.values())
                # import pdb; pdb.set_trace()
                for k, v in episode_images_train.items():
                    if 'z' in k:
                        summary_writer.add_image(k + ' train', v, global_step=step)
                    else:
                        summary_writer.add_image(
                            k + ' train', v, global_step=step, dataformats='HWC')
            # Reset environment if end of episode is reached
            if timestep['done']:
                timestep = self._env.reset()
