from typing import NamedTuple, Optional

from gym import spaces
import numpy as np

from env.env_base import BaseEnv


class Veh3DoFParam(NamedTuple):
    kf: float = -128915.5  # front wheel cornering stiffness [N/rad]
    kr: float = -85943.6   # rear wheel cornering stiffness [N/rad]
    lf: float = 1.06       # distance from CG to front axle [m]
    lr: float = 1.85       # distance from CG to rear axle [m]
    m:  float = 1412.0     # mass [kg]
    Iz: float = 1536.7     # polar moment of inertia at CG [kg*m^2]

    
class Veh3DoF(BaseEnv):
    def __init__(self, seed: Optional[int] = None, max_step: int = 50):
        self.action_space = spaces.Box(
            low=np.array([-np.pi/6, -1.0], dtype=np.float32), 
            high=np.array([np.pi/6, 1.0], dtype=np.float32), 
            shape=(2, ), 
            dtype=np.float32
        )
        # TODO: check the range of the state space
        low = np.array([3, -1, -0.5], dtype=np.float32)
        high = np.array([6, 1, 0.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(3, ), dtype=np.float32)
        self.param = Veh3DoFParam()
        self.dt = 0.1
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
        u, v, w = self.state
        steer, ax = action

        kf = self.param.kf
        kr = self.param.kr
        lf = self.param.lf
        lr = self.param.lr
        m  = self.param.m
        Iz = self.param.Iz
        dt = self.dt

        next_state = self.state.copy()
        next_state[0] = u + dt * ax
        next_state[1] = (
            m * v * u + dt * (lf * kf - lr * kr) * w 
            - dt * kf * steer * u - dt * m * u ** 2 * w
        ) / (m * u - dt * (kf + kr))
        next_state[2] = (
            Iz * w * u + dt * (lf * kf - lr * kr) * v 
            - dt * lf * kf * steer * u
        ) / (Iz * u - dt * (lf ** 2 * kf + lr ** 2 * kr))

        return next_state


    def model_forward(self, state, action):
        unbatched = len(state.shape) == 1
        if unbatched:
            state = state[None, :]
            action = action[None, :]
        u, v, w = state[:, 0], state[:, 1], state[:, 2]
        steer, ax = action[:, 0], action[:, 1]
        next_state = np.stack([
            u + self.dt * ax,
            (
                self.param.m * v * u + self.dt * (self.param.lf * self.param.kf - self.param.lr * self.param.kr) * w 
                - self.dt * self.param.kf * steer * u - self.dt * self.param.m * u ** 2 * w
            ) / (self.param.m * u - self.dt * (self.param.kf + self.param.kr)),
            (
                self.param.Iz * w * u + self.dt * (self.param.lf * self.param.kf - self.param.lr * self.param.kr) * v 
                - self.dt * self.param.lf * self.param.kf * steer * u
            ) / (self.param.Iz * u - self.dt * (self.param.lf ** 2 * self.param.kf + self.param.lr ** 2 * self.param.kr))
        ], axis=1)
        if unbatched:
            next_state = next_state[0]
        return next_state


class Veh3DoFwoControl(Veh3DoF):
    def __init__(self, seed: int = None, max_step: int = 200):
        super().__init__(seed, max_step)
        self.action_space = spaces.Box(low=-0.0, high=0.0, shape=(2, ), dtype=np.float32)

    def get_next_state(self, action):
        return super().get_next_state(np.zeros_like(action))

    def model_forward(self, state, action):
        return super().model_forward(state, np.zeros_like(action))
    