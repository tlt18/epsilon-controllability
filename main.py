from controllabilitytest import ControllabilityTest
from buffer import Buffer
from env.massspring import MassSpring

import numpy as np


if __name__ == "__main__":
    num_sample = 100000
    epsilon = 0.05

    env = MassSpring(seed=1)
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

    test.run(np.array([-0.25, 0.0]))
