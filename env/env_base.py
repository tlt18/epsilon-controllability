from abc import ABCMeta, abstractmethod
from typing import Optional

import gym
from gym.utils import seeding
import numpy as np


class BaseEnv(gym.Env, metaclass=ABCMeta):
    def __init__(self, seed: Optional[int], max_step: int):
        self.max_step = max_step
        self.seed(seed)

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
    
    @abstractmethod
    def step(self, action):
        ...
    
    @abstractmethod
    def done(self):
        ...

    @abstractmethod
    def get_next_state(self, action):
        ...
    
    @abstractmethod
    def model_forward(self, state, action):
        '''
        This function supports batched input and has to be consistent with get_next_state.
        '''
        ...