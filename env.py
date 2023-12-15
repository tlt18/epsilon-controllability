import gym
import numpy as np
from typing import Optional

from gym import spaces
from gym.utils import seeding


class SimpleOCPwithControl(gym.Env):
    def __init__(self, seed: Optional[int] = None, max_step: int = 20):
        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(1, ), dtype=np.float32)
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(2, ), dtype=np.float32)
        self.max_step = max_step
        self.seed(seed)

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

    def reset(self, init_state: Optional[np.ndarray] = None):
        self.step_count = 0
        if init_state is None:
            self.state = self.observation_space.sample()
        else:
            assert init_state.shape == self.observation_space.shape, "The shape of init_state is not correct."
            self.state = init_state
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        return [seed]
    
    def model_forward(self, state, action):
        if len(state.shape) > 1:
            return np.array([1/2 * state[:, 0] + action[:], 1/2 * state[:, 1]], dtype = np.float32).T
        else:
            return np.array([1/2 * state[0] + action, 1/2 * state[1]], dtype = np.float32)
    

class SimpleOCPwoControl(SimpleOCPwithControl, gym.Env):
    def get_next_state(self, action):
        return np.array([1/2 * self.state[0], 1/2 * self.state[1]], dtype = np.float32)
    
    def model_forward(self, state, action):
        if len(state.shape) > 1:
            return np.array([1/2 * state[:, 0], 1/2 * state[:, 1]], dtype = np.float32).T
        else:
            return np.array([1/2 * state[0], 1/2 * state[1]], dtype = np.float32)
    