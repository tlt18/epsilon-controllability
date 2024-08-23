from typing import Optional

from gym import spaces
import numpy as np

from env.env_base import BaseEnv


class CarFollowing(BaseEnv):
    def __init__(self, seed: Optional[int] = None, max_step: int = 30):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1, ), dtype=np.float32)
        # TODO: check the range of the state space
        low = np.array([-1, -1], dtype=np.float32)
        high = np.array([1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(2, ), dtype=np.float32)
        self.param = {
            "k1": 2.0,
            "k2": 3.0,
            "k3": 2.0
        }
        self.dt = 0.1
        super().__init__(seed, max_step)

    def step(self, action):
        self.state = self.get_next_state(action)
        reward = 0
        self.step_count += 1
        done = self.done() or self.step_count >= self.max_step
        done = self.step_count >= self.max_step
        return self.state, reward, done, {}
    
    def done(self):
        if self.observation_space.contains(self.state):
            return False
        else:
            return True

    def get_next_state(self, action):
        
        
        '''
        dynamics:
            \dot{x} = v
            \dot{v} = -k/m * x -\rho/m * v + u/m
        Eluer forward:
            x_{t+1} = x_t + \dot{x}_t * dt
            v_{t+1} = v_t + \dot{v}_t * dt
        discrete-time dynamics:
            x_{t+1} = x_t + \dot{x}_t * dt
            v_{t+1} = v_t + (-k1 * x_t - k2 * v_t + k3 * u) * dt
        '''
        try:
            next_state = np.array([
                self.state[0] + self.state[1] * self.dt, 
                self.state[1] + (-self.param['k1'] * self.state[0] - self.param['k2'] * self.state[1] + self.param['k3'] * action[0]) * self.dt
                ], dtype=np.float32
            )

            # Check for invalid values
            if not np.all(np.isfinite(next_state)):
                raise ValueError(f"Invalid next state encountered: {next_state}")

        except Exception as e:
            print(f"Error in get_next_state: {e}")
            # Handle the error (e.g., return a default state or re-raise the exception)
            next_state = np.array([0.0, 0.0], dtype=np.float32)  # Example default state

        return next_state

    def model_forward(self, state, action):
        unbatched = len(state.shape) == 1
        if unbatched:
            state = state[None, :]
            action = action[None, :]
        next_state = np.stack([
            state[:, 0] + state[:, 1] * self.dt, 
            state[:, 1] + (-self.param['k1'] * state[:, 0] - self.param['k2'] * state[:, 1] + self.param['k3'] * action[:, 0]) * self.dt
            ], axis=1)
        if unbatched:
            next_state = next_state[0]
        return next_state


class CarFollowingwoControl(CarFollowing):
    def __init__(self, seed: int = None, max_step: int = 30):
        super().__init__(seed, max_step)
        self.action_space = spaces.Box(low=-0.0, high=0.0, shape=(1, ), dtype=np.float32)

    def get_next_state(self, action):
        return np.array([
            self.state[0] + self.state[1] * self.dt, 
            self.state[1] + (-self.param['k1'] * self.state[0] - self.param['k2'] * self.state[1]) * self.dt
            ], dtype = np.float32
        )

    def model_forward(self, state, action):
        unbatched = len(state.shape) == 1
        if unbatched:
            state = state[None, :]
        next_state = np.stack([
            state[:, 0] + state[:, 1] * self.dt, 
            state[:, 1] + (-self.param['k1'] * state[:, 0] - self.param['k2'] * state[:, 1]) * self.dt
            ], axis=1)
        if unbatched:
            next_state = next_state[0]
        return next_state