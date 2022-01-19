"""Tester class to evaluate models after training"""

from torch.utils import tensorboard 
import os
class Tester():
    """Tested class."""

    def __init__(self, 
                agent,
                env,
                iterations):
        """Constructor

        Args:
            agent (object): Instance of agent. Must have methods:
                * step(self, timestep, env, test=False) returning action.
                * train_step(self, prev_timestep, action, timestep) returning
                    loss.
            env (object): Instance of environment used for training
            iterations (int): Number of testing iterations        
        """
        self._agent = agent
        self._env = env
        self._iterations = iterations

    def __call__(self, log_dir):
        """Run a testing loop.

        Args:
            log_dir (string): Directory in which to log tensorboard logs for scalars and images
        """
        
        summary_dir = os.path.join(log_dir, 'tensorboard') 
        summary_writer = tensorboard.SummaryWriter(log_dir=summary_dir)

        test_figure = self._env.generate_test_figure(self._agent, max_steps=self._iterations) 
        for k, v in test_figure.items():
            if 'z' in k:
                summary_writer.add_image(k + ' test_figure', v, global_step=20000)
            else:
                summary_writer.add_image(
                    k + ' test_figure', v, global_step=200000, dataformats='HWC')

        episode_image_figure = self._env.generate_episode_figure(self._agent, max_steps=self._iterations) 
        for k, v in episode_image_figure.items():
            if 'z' in k:
                summary_writer.add_image(k + ' episode_test_figure', v, global_step=20000)
            else:
                summary_writer.add_image(
                    k + ' episode_test_figure', v, global_step=200000, dataformats='HWC')

        summary_writer.close()

