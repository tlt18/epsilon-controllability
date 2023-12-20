from buffer import Buffer
from env.massspring import MassSpring, MassSpringwoControl
from env.simpleocp import SimpleOCP, SimpleOCPwoControl
from env.pendulum import Pendulum, PendulumwoControl
from controllabilitytest import ControllabilityTest, Transition
from utils.timeit import Timeit

import numpy as np


if __name__ == "__main__":
    num_sample = 100000
    epsilon = 0.05
    target_state = None
    lipschitz_confidence = 0.2

    env = Pendulum(seed=1)
    buffer = Buffer(buffer_size = num_sample)
    test = ControllabilityTest(
        env = env,
        buffer = buffer,
        target_state = target_state,
        epsilon = epsilon, 
        num_sample = num_sample,
        lipschitz_confidence = lipschitz_confidence,
        use_kd_tree = True,
        lips_estimate_mode = "sampling",
        expand_plot_interval = 1000, 
        backward_plot_interval = 10000000000,
        plot_expand_flag = False,
        plot_backward_flag = False,
    )
    test.sample()

    for _ in range(5):
        state = env.observation_space.sample()
        action = env.action_space.sample()
        next_state = env.model_forward(state, action)
        transition = Transition(state, action, next_state)
        with Timeit("lipschitz_fx_sampling"):
            lips_by_sample = test.lipschitz_fx_sampling(transition)
        with Timeit("lipschitz_fx_optimizing"):
            lips_by_opt = test.lipschitz_fx_optimizing(transition)
        with Timeit("lipschitz_fx_overestimating"):
            lips_by_maxdist = test.lipschitz_fx_maxdistance(transition)
        print("-" * 50)
        print(f"lips_by_sample: {lips_by_sample}, lips_by_opt: {lips_by_opt}, lips_by_maxdist: {lips_by_maxdist}")
