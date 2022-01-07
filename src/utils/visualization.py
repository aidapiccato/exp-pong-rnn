import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from envs.exp_pong_env import GRID_DIM



def _image_reshaping(images, buffer_height):
    height, width, channels = images[0].shape
    num_steps = len(images)
    images = np.array(images)
    buffer = np.zeros(
        (num_steps, buffer_height, width, channels), dtype=np.uint8)
    images = np.concatenate((images, buffer), axis=1)
    images = np.reshape(
        images, (num_steps * (height + buffer_height), width, channels))
    # images = np.reshape(
        # images, (1, num_steps, channels, height, width))

    return images

def _generate_episode_figure(agent_pos):
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()        
        ax.plot(agent_pos)
        ax.set_ylim(-1, GRID_DIM)
        ax.set_ylabel('agent position')
        ax.set_xlabel('timestep')
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas.draw()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        return [image]

def generate_episode_figure(env, agent, buffer_height=3, max_steps=10):
    num_steps = 0
    agent_pos = []
    actions = []
    while num_steps == 0:
        timestep = env.reset()
        agent_pos = []
        actions = []
        while not timestep['done'] and num_steps < max_steps:
            num_steps += 1
            action = agent.step(timestep, env, test=True)            
            actions.append(action)
            timestep = env.step(action)
            agent_pos.append(env.agent_pos)
        agent_pos = np.array(agent_pos).squeeze()
        image = _generate_episode_figure(agent_pos)
    images_dict = {'validation': _image_reshaping(image, buffer_height)}    
    return images_dict

def generate_episode_video(env, agent, buffer_height=3, max_steps=10):
    """Generate a video of an episode."""

    num_steps = 0
    images = []
    while num_steps == 0:
        timestep = env.reset()
        images = []

        while not timestep['done'] and num_steps < max_steps:
            num_steps += 1
            image = timestep['image']
            action = agent.step(timestep, env, test=True)
            # _render_action(image, action, env.action_space)
            images.append(image)
            timestep = env.step(action)
    images_dict = {'validation': _image_reshaping(images, buffer_height)}    
    return images_dict