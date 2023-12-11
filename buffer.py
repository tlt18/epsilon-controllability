# replay buffer for off-line training
import numpy as np
import random
from collections import deque
from typing import Tuple


class Buffer:
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_idx = 0

    def add(self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]):
        # transition: (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert batch_size <= len(self.buffer), "The batch size is larger than the buffer size."
        batch = random.sample(self.buffer, batch_size)
        state_batch = np.array([transition[0] for transition in batch])
        action_batch = np.array([transition[1] for transition in batch])
        reward_batch = np.array([transition[2] for transition in batch])
        next_state_batch = np.array([transition[3] for transition in batch])
        done_batch = np.array([transition[4] for transition in batch])
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
        self.buffer_idx = 0