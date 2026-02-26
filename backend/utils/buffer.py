from collections import deque
import numpy as np

class LandmarkBuffer:
    def __init__(self, maxlen=30):
        self.buffer = deque(maxlen=maxlen)

    def add(self, landmarks):
        """Appends one frame of landmarks to the buffer."""
        self.buffer.append(landmarks)

    def is_ready(self):
        """Returns True when the buffer has reached its target length."""
        return len(self.buffer) == self.buffer.maxlen

    def get_sequence(self):
        """Returns the current buffer content as a list of frames."""
        return list(self.buffer)

    def clear(self):
        """Resets the buffer."""
        self.buffer.clear()

    def size(self):
        """Returns the current number of frames in the buffer."""
        return len(self.buffer)
