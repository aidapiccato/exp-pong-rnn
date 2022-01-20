import os
import logging
from torch.utils import tensorboard
import torch
from envs.batch_env import BatchEnv

class EpisodeTrainer:
    """Trainer class for trial-based updates. 
    """

    def __init__(self,
                 agent, 
                 batch_size,
                 env_class,
                 env_kwargs,
                 episodes,
                 scalar_eval_every, 
                 snapshot_every,
                 image_eval_every):
        """Constructor. 
        """
        self._agent = agent
        self._env_class = env_class
        self._env_kwargs = env_kwargs
        self._batch_size = batch_size
        self._episodes = episodes 
        self._scalar_eval_every = scalar_eval_every
        self._snapshot_every = snapshot_every
        self._image_eval_every = image_eval_every


    def __call__(self, log_dir):
        """Run a training loop

        Args:
            log_dir ([type]): [description]
        """
        env_batch = BatchEnv(batch_size=self._batch_size, env_class=self._env_class, **self._env_kwargs)
        agg_reward = 0.
        agg_loss = 0.

        snapshot_dir = os.path.join(log_dir, 'snapshots')
        os.makedirs(snapshot_dir)
        summary_dir = os.path.join(log_dir, 'tensorboard')
        summary_writer = tensorboard.SummaryWriter(log_dir=summary_dir)

        for episode in range(self._episodes):
            training_output = self._agent.train_episode(env_batch)
            agg_loss += training_output['loss']
            agg_reward += training_output['reward']

            if episode % self._scalar_eval_every == 0:
                scalars = {
                    'episode': episode,
                    'train_reward': agg_reward / self._scalar_eval_every,
                    'loss': agg_loss / self._scalar_eval_every,
                }
                # Log scalars
                logging.info(scalars)
                for k, v in scalars.items():
                    summary_writer.add_scalar(k, v, global_step=episode)
                agg_reward = 0.
                agg_loss = 0.

            if episode % self._image_eval_every == 0:
                logging.debug('Generating figure')
                episode_image_figure = env_batch.generate_episode_figure(agent=self._agent, max_steps=100)
                for k, v in episode_image_figure.items():
                    if 'z' in k:
                        summary_writer.add_image(k + ' train_figure', v, global_step=episode)
                    else:
                        summary_writer.add_image(
                            k + ' train_figure', v, global_step=episode, dataformats='HWC')

            
            if episode % self._snapshot_every == 0:
                logging.info('Saving snapshot.')
                snapshot_filename = os.path.join(snapshot_dir, str(episode))
                torch.save(self._agent.state_dict(), snapshot_filename)

