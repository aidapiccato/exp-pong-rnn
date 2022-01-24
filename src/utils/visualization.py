from importlib_metadata import itertools
import seaborn as sns
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


def _DFS(G, v, seen=None, path=None):
    if seen is None:
        seen = []
    if path is None:
        path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(_DFS(G, t, seen[:], t_path))
    return paths


def _find_optimal_reward(target_x, target_t):
    n = len(target_x)
    G = {k: [] for k in range(n)}
    for i, j in itertools.combinations(range(n), 2):
        if (target_t[j] - target_t[i]) >= np.abs((target_x[j] - target_x[i])):
            G[i].append(j)

    paths = _DFS(G, 0)
    if len(paths) == 0:
        max_len_path = 0
    else:
        max_len_path = np.amax([len(path) for path in paths])
    return max_len_path


def generate_summary_figure(agent, max_episodes, buffer_height=3):
    rewards = np.array([])
    optimal_rewards = np.array([])
    for _ in range(max_episodes):
        outputs = agent.forward(test=True)
        episode_rewards = np.stack(outputs['rewards'])
        episode_infos = outputs['infos']
        episode_infos = [{k: [d[k] for d in info]
                          for k in episode_infos[0][0].keys()} for info in episode_infos]
        episode_infos = {k: [d[k] for d in episode_infos]
                         for k in episode_infos[0].keys()}

        target_xs = episode_infos['target_x'][0]
        target_ts = episode_infos['target_t'][0]
        optimal_rewards = np.concatenate((optimal_rewards, [_find_optimal_reward(
            target_x, target_t) for target_x, target_t in zip(target_xs, target_ts)]))
        rewards = np.concatenate((rewards, np.sum(episode_rewards, axis=0)))

    fig = Figure(figsize=(5, 5))
    canvas = FigureCanvas(fig)

    ax = fig.add_subplot(111)
    sns.regplot(x=optimal_rewards, y=rewards, ax=ax)
    mn = min(optimal_rewards.min(), rewards.min())
    mx = max(optimal_rewards.max(), rewards.max())
    points = np.linspace(mn, mx, 100)
    ax.plot(points, points, color='k', marker=None,
            linestyle='--', linewidth=1.0)
    fig.tight_layout()

    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas.draw()

    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(
        int(height), int(width), 3)

    image_dict = {'validation_summary': image_reshaping([image], buffer_height)}
    return image_dict


def generate_episode_figure(agent, max_steps, buffer_height=3):
    outputs = agent.forward(test=True)
    observations = np.stack(outputs['observations'])
    actions = np.stack(outputs['actions'])
    rewards = np.stack(outputs['rewards'])
    infos = outputs['infos']
    infos = [{k: [d[k] for d in info] for k in infos[0][0].keys()}
             for info in infos]
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
