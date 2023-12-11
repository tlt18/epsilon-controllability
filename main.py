import numpy as np
from controllabilitytest import ControllabilityTest
from buffer import Buffer
from env import SimpleOCPwithControl, SimpleOCPwoControl

import time

if __name__ == "__main__":
    num_sample = 100
    epsilon = 0.5

    env = SimpleOCPwoControl(seed=1, max_step=20)
    buffer = Buffer(buffer_size = num_sample)
    test = ControllabilityTest(env = env, buffer = buffer, num_sample = num_sample, epsilon = epsilon, lipschitz_confidence = np.inf)

    time_start = time.time()
    test.sample()
    time_sample = time.time() - time_start
    print("time for sampling: {:.4f}s".format(time_sample))

    test.get_epsilon_controllable_set(env.observation_space.sample())
    time_calonestep = time.time() - time_start - time_sample
    print("time for calculating one-step matrix: {:.4f}s".format(time_calonestep))

    print(test.epsilon_controllable_list)
