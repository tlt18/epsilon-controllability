from typing import Optional

from gym import spaces
import numpy as np

from env.env_base import BaseEnv


class TunnelDiode(BaseEnv):
    def __init__(self, seed: Optional[int] = None, max_step: int = 50):
        self.action_space = spaces.Box(low=-0.0, high=0.0, shape=(1, ), dtype=np.float32)
        low = np.array([-0.3, -0.3], dtype=np.float32)
        high = np.array([1.4, 1.4], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(2, ), dtype=np.float32)
        self.param = {
            'u': 1.2,
            'R': 1.5,
            'C': 2.0,
            'L': 5.0
        }
        self.dt = 0.1
        super().__init__(seed, max_step)

    def step(self, action):
        last_state = self.state
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
        https://courses.engr.illinois.edu/ece486/fa2023/documentation/handbook/lec02.html
        Dynamics:
            \dot{x} = 1/C * (-h(x) + y)
            \dot{y} = 1/L * (- x - R * y + u)
        Eluer forward:
            x_{t+1} = x_t + \dot{x}_t * dt
            y_{t+1} = y_t + \dot{y}_t * dt
        '''
        
        dot_x = 1 / self.param['C'] * (-self.h(self.state[0]) + self.state[1])
        dot_y = 1 / self.param['L'] * (-self.state[0] - self.param['R'] * self.state[1] + self.param['u'])
        dot_x = np.clip(dot_x, -1, 1)
        dot_y = np.clip(dot_y, -1, 1)

        return np.array([
            self.state[0] + dot_x * self.dt, 
            self.state[1] + dot_y * self.dt
            ], dtype = np.float32
        )
    
    def h(self, x):
        return 17.76 * x - 103.79 * x**2 + 229.62 * x**3 - 226.31 * x**4 + 83.72 * x**5

    def model_forward(self, state, action):
        unbatched = len(state.shape) == 1
        if unbatched:
            state = state[None, :]
            action = action[None, :]
        dot_x = 1 / self.param['C'] * (-self.h(state[:, 0]) + state[:, 1])
        dot_y = 1 / self.param['L'] * (-state[:, 0] - self.param['R'] * state[:, 1] + self.param['u'])
        dot_x = np.clip(dot_x, -1, 1)
        dot_y = np.clip(dot_y, -1, 1)
        next_state = np.stack([
            state[:, 0] + dot_x * self.dt, 
            state[:, 1] + dot_y * self.dt
            ], axis=1)
        if unbatched:
            next_state = next_state[0]
        return next_state

if __name__ == "__main__":
    env = TunnelDiode(seed = 1)
    # state = np.array([0.8844298, 0.210380361])
    # state = np.array([0.06263583 0.75824183])
    # print(env.model_forward(state, np.array([0.0])))

    for _ in range(10):
        print("Reset")
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                print(state, reward, done, env.step_count)

