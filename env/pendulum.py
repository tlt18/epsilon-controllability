from typing import Optional

from gym import spaces
import numpy as np

from env.env_base import BaseEnv


class Pendulum(BaseEnv):
    def __init__(self, seed: Optional[int] = None, max_step: int = 30):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1, ), dtype=np.float32)
        low = np.array([-0.5, -0.5], dtype=np.float32)
        high = np.array([0.5, 0.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(2, ), dtype=np.float32)
        self.dt = 0.1
        self.param = {
            "g": 9.8,
            "m": 1,
            "l": 1,
            "f": 0.1
        }
        super().__init__(seed = seed, max_step = max_step)

    def step(self, action):
        self.state = self.get_next_state(action)
        reward = 0
        self.step_count += 1
        done = self.step_count >= self.max_step
        return self.state, reward, done, {}
    
    def done(self):
        if self.observation_space.contains(self.state):
            return False
        else:
            return True

    def get_next_state(self, action):
        '''
        Dynamics:
            \dot{\theta} = \omega
            \dot{\omega} = -g/l * \sin(\theta) - f/m * \omega + u/(ml^2)
        Eluer forward:
            \theta_{t+1} = \theta_t + \dot{\theta}_t * dt
            \omega_{t+1} = v_t + \dot{\omega}_t * dt
        '''
        return np.array([
            self.state[0] + self.state[1] * self.dt, 
            self.state[1] + (- self.param["g"] / self.param["l"] * np.sin(self.state[0]) - \
                self.param["f"] / self.param["m"] * self.state[1] + \
                action[0] / (self.param["m"] * self.param["l"] ** 2)) * self.dt
            ], dtype = np.float32
        )

    def model_forward(self, state, action):
        unbatched = len(state.shape) == 1
        if unbatched:
            state = state[None, :]
            action = action[None, :]
        next_state = np.stack([
            state[:, 0] + state[:, 1] * self.dt, 
            state[:, 1] + (- self.param["g"] / self.param["l"] * np.sin(state[:, 0]) - \
                self.param["f"] / self.param["m"] * state[:, 1] + \
                action[:, 0] / (self.param["m"] * self.param["l"] ** 2)) * self.dt
            ], axis=1)
        if unbatched:
            next_state = next_state[0]
        return next_state


class PendulumwoControl(Pendulum):
    def __init__(self, seed: int = None, max_step: int = 30):
        super().__init__(seed, max_step)
        self.action_space = spaces.Box(low=-0.0, high=0.0, shape=(1, ), dtype=np.float32)

    def get_next_state(self, action):
        '''
        Dynamics:
            \dot{\theta} = \omega
            \dot{\omega} = -g/l * \sin(\theta) - f/m * \omega
        Eluer forward:
            \theta_{t+1} = \theta_t + \dot{\theta}_t * dt
            \omega_{t+1} = v_t + \dot{\omega}_t * dt
        '''
        return np.array([
            self.state[0] + self.state[1] * self.dt, 
            self.state[1] + (- self.param["g"] / self.param["l"] * np.sin(self.state[0]) - \
                self.param["f"] / self.param["m"] * self.state[1]) * self.dt
            ], dtype = np.float32
        )
    
    def model_forward(self, state, action):
        unbatched = len(state.shape) == 1
        if unbatched:
            state = state[None, :]
        next_state = np.stack([
            state[:, 0] + state[:, 1] * self.dt, 
            state[:, 1] + (- self.param["g"] / self.param["l"] * np.sin(state[:, 0]) - \
                self.param["f"] / self.param["m"] * state[:, 1]) * self.dt
            ], axis=1)
        if unbatched:
            next_state = next_state[0]
        return next_state
    