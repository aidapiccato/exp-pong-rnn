"""Replay buffers."""

import numpy as np


class FIFO():
    """First-in, first-out replay buffer.
    
    Usage:
        ```python
        replay = FIFO(capacity=int(1e3))
        replay.write(x1, y1, z1)
        replay.write(x2, y2, z2)
        ...
        x_batch, y_batch, z_batch = replay.read(batch_size=N)
        ```
    """

    def __init__(self, capacity=int(1e4)):
        """Constructor.
        Args:
            capacity: Int. Maximum capacity of the replay buffer.
        """
        self._capacity = capacity
        self._replay = []

    def read(self, batch_size):
        """Read a batch of data from the replay buffer.
        Does not remove the read data from the buffer.
        Args:
            batch_size: Int. Batch size.
        Returns:
            data batch iterable of elements, one for each of the elements
                written to the replay. Each has shape [batch_size, ...].
        """
        read_inds = np.random.randint(len(self._replay), size=(batch_size,))
        data_batch = [self._replay[i] for i in read_inds]

        data_batch = [np.array(x) for x in zip(*data_batch)]
        return data_batch

    def write(self, *data):
        """Write data to replay."""
        while len(self._replay) >= self._capacity:
            self._replay.pop(0)
        self._replay.append(data)