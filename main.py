from controllabilitytest import ControllabilityTest
from buffer import Buffer
from env.massspring import MassSpring, MassSpringwoControl
from env.simpleocp import SimpleOCP, SimpleOCPwoControl
from env.pendulum import Pendulum, PendulumwoControl

import numpy as np


if __name__ == "__main__":
    num_sample = 100000
    epsilon = 0.05
    target_state = np.array([-0.0, 0.0])
    lipschitz_confidence = 0.2

    env = MassSpring(seed=1)
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
        expand_plot_interval = 5000, 
        backward_plot_interval = 10000000000,
        plot_expand_flag = True,
        plot_backward_flag = False,
    )

    test.run()
