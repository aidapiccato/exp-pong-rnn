from torch.utils import tensorboard
import os
import utils.visualization


class Tester():
    """Tested class."""

    def __init__(self,
                 agent,
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
        self._iterations = iterations

    def __call__(self, log_dir):
        """Run a testing loop.
        Args:
            log_dir (string): Directory in which to log tensorboard logs for scalars and images
        """

        summary_dir = os.path.join(log_dir, 'tensorboard')
        summary_writer = tensorboard.SummaryWriter(log_dir=summary_dir)

        episode_images_test = utils.visualization.generate_episode_figure(
            self._agent, max_steps=60)
        for k, v in episode_images_test.items():
            if 'z' in k:
                summary_writer.add_image(
                    k + ' test', v, global_step=0)
            else:
                summary_writer.add_image(
                    k + ' test', v, global_step=0, dataformats='HWC')

        summary_figure_test = utils.visualization.generate_summary_figure(
            self._agent, max_episodes=self._iterations)
        for k, v in summary_figure_test.items():
            if 'z' in k:
                summary_writer.add_image(
                    k + ' test', v, global_step=0)
            else:
                summary_writer.add_image(
                    k + ' test', v, global_step=0, dataformats='HWC')

        summary_writer.close()
