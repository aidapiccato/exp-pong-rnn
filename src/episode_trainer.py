"""Trainer class for RL agents."""

import os
import logging
from torch.utils import tensorboard
import torch
import utils.visualization


class EpisodeTrainer:
    """Trainer class for trial-based updates. """

    def __init__(self,
                 agent,
                 episodes,
                 scalar_eval_every,
                 snapshot_every,
                 image_eval_every):
        """Constructor. 
        """
        self._agent = agent
        self._episodes = episodes
        self._scalar_eval_every = scalar_eval_every
        self._snapshot_every = snapshot_every
        self._image_eval_every = image_eval_every

    def __call__(self, log_dir):
        """Run a training loop

        Args:
            log_dir ([type]): [description]
        """
        snapshot_dir = os.path.join(log_dir, 'snapshots')
        os.makedirs(snapshot_dir)
        summary_dir = os.path.join(log_dir, 'tensorboard')
        summary_writer = tensorboard.SummaryWriter(log_dir=summary_dir)

        for episode in range(self._episodes):
            self._agent.train()

            if episode % self._scalar_eval_every == 0:
                scalars = self._agent.scalars()
                scalars['episode'] = episode
                logging.info(scalars)
                for k, v in scalars.items():
                    summary_writer.add_scalar(k, v, global_step=episode)

            if episode % self._image_eval_every == 0:
                episode_images_train = utils.visualization.generate_episode_figure(
                    self._agent, max_steps=60)
                for k, v in episode_images_train.items():
                    if 'z' in k:
                        summary_writer.add_image(
                            k + ' train', v, global_step=episode)
                    else:
                        summary_writer.add_image(
                            k + ' train', v, global_step=episode, dataformats='HWC')

            if episode % self._snapshot_every == 0:
                logging.info('Saving snapshot.')
                snapshot_filename = os.path.join(snapshot_dir, str(episode))
                torch.save(self._agent.state_dict(), snapshot_filename)
