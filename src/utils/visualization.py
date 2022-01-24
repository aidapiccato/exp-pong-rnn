import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def image_reshaping(images, buffer_height):
    height, width, channels = images[0].shape
    num_steps = len(images)
    images = np.array(images)
    buffer = np.zeros(
        (num_steps, buffer_height, width, channels), dtype=np.uint8)
    images = np.concatenate((images, buffer), axis=1)
    images = np.reshape(
        images, (num_steps * (height + buffer_height), width, channels))
    return images


def generate_episode_figure(agent, max_steps, buffer_height=3):
    outputs = agent.forward(test=True)
    observations = np.stack(outputs['observations'])
    actions = np.stack(outputs['actions'])
    rewards = np.stack(outputs['rewards'])
    infos = outputs['infos']
    infos = [{k: [d[k] for d in info] for k in infos[0][0].keys()} for info in infos]
    infos = {k: [d[k] for d in infos] for k in infos[0].keys()}


    rewards = rewards[:, 0]
    actions = actions[:, 0]
    agent_pos = np.stack(infos['agent_pos'])[:, 0]
    target_x = infos['target_x'][0][0]
    target_t = infos['target_t'][0][0]

    observations = observations[:, 0, :]

    fig = Figure(figsize=(8, 4))
    canvas = FigureCanvas(fig)

    ax = fig.add_subplot(121)
    ax_right = ax.twinx()
    ax_right.plot(rewards, c='red')

    ax_right.set_ylabel('reward')
    ax.plot(agent_pos)
    ax.set_ylabel('agent')
    ax.scatter(target_t, target_x, c='green')

    ax = fig.add_subplot(122)
    ax.imshow(observations)
    ax.set_xlabel('x')
    ax.set_ylabel('time')

    fig.tight_layout()

    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas.draw()

    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(
        int(height), int(width), 3)

    image_dict = {'validation': image_reshaping([image], buffer_height)}
    return image_dict
