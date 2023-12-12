from controllabilitytest import ControllabilityTest
from buffer import Buffer
from env import SimpleOCPwithControl, SimpleOCPwoControl

import numpy as np
import time


if __name__ == "__main__":
    num_sample = 100
    epsilon = 0.1

    env = SimpleOCPwithControl(seed=1, max_step=5)
    buffer = Buffer(buffer_size = num_sample)
    test = ControllabilityTest(env = env, buffer = buffer, num_sample = num_sample, epsilon = epsilon, lipschitz_confidence = 1)

    test.run(np.zeros_like(env.observation_space.sample()))
