from typing import Optional

from gym import spaces
import numpy as np

from env.env_base import BaseEnv


class MassSpring(BaseEnv):
    def __init__(self, seed: Optional[int] = None, max_step: int = 200):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1, ), dtype=np.float32)
        # TODO: check the range of the state space
        low = np.array([-1, -1], dtype=np.float32)
        high = np.array([1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(2, ), dtype=np.float32)
        self.param = {
            'm': 1.0,
            'k': 1.0,
            'rho': 0.8,
        }
        self.dt = 0.1
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
        '''
        Dynamics:
            \dot{x} = v
            \dot{v} = -k/m * x -\rho/m * v + u/m
        Eluer forward:
            x_{t+1} = x_t + \dot{x}_t * dt
            v_{t+1} = v_t + \dot{v}_t * dt
        '''
        return np.array([
            self.state[0] + self.state[1] * self.dt, 
            self.state[1] + (-self.param['k'] * self.state[0] - self.param['rho'] * self.state[1] + action[0]) / self.param['m'] * self.dt
            ], dtype = np.float32
        )

    def model_forward(self, state, action):
        unbatched = len(state.shape) == 1
        if unbatched:
            state = state[None, :]
            action = action[None, :]
        next_state = np.stack([
            state[:, 0] + state[:, 1] * self.dt, 
            state[:, 1] + (-self.param['k'] * state[:, 0] - self.param['rho'] * state[:, 1] + action[:, 0]) / self.param['m'] * self.dt
            ], axis=1)
        if unbatched:
            next_state = next_state[0]
        return next_state


class MassSpringwoControl(MassSpring):
    def __init__(self, seed: int = None, max_step: int = 200):
        super().__init__(seed, max_step)
        self.action_space = spaces.Box(low=-0.0, high=0.0, shape=(1, ), dtype=np.float32)

    def get_next_state(self, action):
        return np.array([
            self.state[0] + self.state[1] * self.dt, 
            self.state[1] + (-self.param['k'] * self.state[0] - self.param['rho'] * self.state[1]) / self.param['m'] * self.dt
            ], dtype = np.float32
        )

    def model_forward(self, state, action):
        unbatched = len(state.shape) == 1
        if unbatched:
            state = state[None, :]
        next_state = np.stack([
            state[:, 0] + state[:, 1] * self.dt, 
            state[:, 1] + (-self.param['k'] * state[:, 0] - self.param['rho'] * state[:, 1]) / self.param['m'] * self.dt
            ], axis=1)
        if unbatched:
            next_state = next_state[0]
        return next_state