from typing import NamedTuple, Optional

from gym import spaces
import numpy as np

from env.env_base import BaseEnv


class LorenzParam(NamedTuple):

    sigma: float = 10
    rho: float = 28
    beta: float = 8/3


    
class Lorenz(BaseEnv):
    def __init__(self, seed: Optional[int] = None, max_step: int = 50):
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([0, 0], dtype=np.float32),
            shape=(2, ), 
            dtype=np.float32
        )
        # TODO: check the range of the state space
        low = np.array([-30, -30, 0], dtype=np.float32)
        high = np.array([30, 30, 60], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(3, ), dtype=np.float32)
        self.param = LorenzParam()
        self.dt = 0.01
        super().__init__(seed, max_step)

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

    def get_next_state(self, action: np.ndarray) -> np.ndarray:
        x, y, z = self.state
        steer, ax = action

        sigma = self.param.sigma
        rho = self.param.rho
        beta = self.param.beta

        dt = self.dt

        next_state = self.state.copy()
        next_state[0] = x + dt * (sigma*(y-x))
        next_state[1] = y + dt*(x*(rho-z)-y)

        next_state[2] = z + dt * (x*y-beta*z)

        return next_state


    def model_forward(self, state, action):
        unbatched = len(state.shape) == 1
        if unbatched:
            state = state[None, :]
            action = action[None, :]
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        steer, ax = action[:, 0], action[:, 1]
        next_state = np.stack([
            x + self.dt * (self.param.sigma*(y - x)),
            y + self.dt*(x*(self.param.rho-z)-y),
            z + self.dt*(x*y-self.param.beta*z)
        ], axis=1)
        if unbatched:
            next_state = next_state[0]
        return next_state


class LorenzwoControl(Lorenz):
    def __init__(self, seed: int = None, max_step: int = 200):
        super().__init__(seed, max_step)
        self.action_space = spaces.Box(low=-0.0, high=0.0, shape=(2, ), dtype=np.float32)

    def get_next_state(self, action):
        return super().get_next_state(np.zeros_like(action))

    def model_forward(self, state, action):
        return super().model_forward(state, np.zeros_like(action))
    