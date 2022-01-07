import numpy as np
def _render_action(image, action, action_space):
    """Render a grid action onto an image.
    
    This function creates a red border on the image in the direction of the
    action, if the action space is oog.action_spaces.Grid.
    Args:
        image: Uint8 Numpy array of shape [height, width, channels]. Image
            observation from environment timestep.
        action: Action produced by the agent.
        action_space: Action spaces. Should be env.action_space.
    """

    if action == 0:
        image[:, :1] = np.array([[255, 0, 0]], dtype=np.uint8)
    elif action == -1:
        image[:, -1:] = np.array([[255, 0, 0]], dtype=np.uint8)
    elif action == 1:
        image[-1:] = np.array([[255, 0, 0]], dtype=np.uint8)

def _image_reshaping(images, buffer_height):
    # import pdb; pdb.set_trace()
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