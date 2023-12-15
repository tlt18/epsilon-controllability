from controllabilitytest import ControllabilityTest
from buffer import Buffer
from env import SimpleOCPwithControl, SimpleOCPwoControl

import numpy as np


if __name__ == "__main__":
    num_sample = 100000
    epsilon = 0.1

    env = SimpleOCPwithControl(seed=1, max_step=5)
    buffer = Buffer(buffer_size = num_sample)
    test = ControllabilityTest(
        env = env,
        buffer = buffer,
        num_sample = num_sample,
        epsilon = epsilon, 
        lipschitz_confidence = 1,
        use_kd_tree = True,
        expand_plot_interval = 10, 
        backward_plot_interval = 100,
        plot_flag = False,
    )

    test.run(np.zeros_like(env.observation_space.sample()))
