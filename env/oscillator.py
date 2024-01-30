from typing import Optional

from gym import spaces
import numpy as np

from env.env_base import BaseEnv


class Oscillator(BaseEnv):
    def __init__(self, seed: Optional[int] = None, max_step: int = 30):
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)
        low = np.array([-1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)
        self.dt = 0.1

        super().__init__(seed=seed, max_step=max_step)

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
            \dot{\theta} = \omega
            \dot{\omega} = -\theta -\omega(1-\theta**2)/2+u
        Eluer forward:
            \theta_{t+1} = \theta_t + \dot{\theta}_t * dt
            \omega_{t+1} = v_t + \dot{\omega}_t * dt
        '''
        u = action
        return np.array([
            self.state[0] + self.state[1] * self.dt,
            self.state[1] + (-self.state[0]-self.state[1](1-self.state[0]**2)/2+ u) * self.dt
        ], dtype=np.float32
        )

    def model_forward(self, state, action):
        unbatched = len(state.shape) == 1
        if unbatched:
            state = state[None, :]
            action = action[None, :]
        next_state = np.stack([
            state[:, 0] + state[:, 1] * self.dt,
            state[:, 1] + (-state[:, 0]-state[:, 1](1-state[:, 0]**2)/2+ action) * self.dt], axis=1)
        if unbatched:
            next_state = next_state[0]
        return next_state


class OscillatorControl(Oscillator):
    def __init__(self, seed: int = None, max_step: int = 30):
        super().__init__(seed, max_step)
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)

    def get_next_state(self, action):
        return super().get_next_state(self.state, np.zeros_like(action))

    def model_forward(self, state, action):
        return super().model_forward(state, np.zeros_like(action))
