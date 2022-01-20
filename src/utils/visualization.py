import numpy as np 

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

