from buffer import Buffer
from env.massspring import MassSpring, MassSpringwoControl
from env.simpleocp import SimpleOCP, SimpleOCPwoControl
from env.pendulum import Pendulum, PendulumwoControl
from controllabilitytest import ControllabilityTest, Transition

import numpy as np


if __name__ == "__main__":
    num_sample = 10000
    epsilon = 0.05

    env = Pendulum(seed=1)
    buffer = Buffer(buffer_size = num_sample)
    test = ControllabilityTest(
        env = env,
        buffer = buffer,
        num_sample = num_sample,
        epsilon = epsilon, 
        lipschitz_confidence = 0.2,
        use_kd_tree = True,
        expand_plot_interval = 1000, 
        backward_plot_interval = 10000000000,
        plot_flag = True,
    )
    test.sample()

    for _ in range(100):
        state = env.observation_space.sample()
        action = env.action_space.sample()
        next_state = env.model_forward(state, action)
        transition = Transition(
            state = state,
            action = action,
            next_state = next_state,
        )
        lips_by_sample = test.lipschitz_fx_sampling(state)
        lips_by_opt = test.lipschitz_fx(transition)
        print("-" * 50)
        print(f"lips_by_sample: {lips_by_sample}, lips_by_opt: {lips_by_opt}")
