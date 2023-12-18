from typing import Optional

from gym import spaces
import numpy as np

from env.env_base import BaseEnv


class SimpleOCP(BaseEnv):
    def __init__(self, seed: Optional[int] = None, max_step: int = 20):
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(1, ), dtype=np.float32)
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(2, ), dtype=np.float32)
        super().__init__(seed, max_step)

    def step(self, action):
        self.state = self.get_next_state(action)
        reward = 0
        self.step_count += 1
        done = self.done() or self.step_count >= self.max_step
        return self.state, reward, done, {}
    
    def done(self):
        if self.observation_space.contains(self.state):
            return False
        else:
            return True

    def get_next_state(self, action):
        return np.array([1/2 * self.state[0] + action[0], 1/2 * self.state[1]], dtype = np.float32)

    def model_forward(self, state, action):
        unbatched = len(state.shape) == 1
        if unbatched:
            state = state[None, :]
            action = action[None, :]
        next_state = np.array([1/2 * state[:, 0] + action[:, 0], 1/2 * state[:, 1]], dtype = np.float32).T
        if unbatched:
            next_state = next_state[0]
        return next_state


class SimpleOCPwoControl(SimpleOCP):
    def __init__(self, seed: int = None, max_step: int = 20):
        super().__init__(seed, max_step)
        self.action_space = spaces.Box(low=-0.0, high=0.0, shape=(1, ), dtype=np.float32)

    def get_next_state(self, action):
        return np.array([1/2 * self.state[0], 1/2 * self.state[1]], dtype = np.float32)
    
    def model_forward(self, state, action):
        unbatched = len(state.shape) == 1
        if unbatched:
            state = state[None, :]
        next_state = np.array([1/2 * state[:, 0], 1/2 * state[:, 1]], dtype = np.float32).T
        if unbatched:
            next_state = next_state[0]
        return next_state
    