from controllabilitytest import ControllabilityTest
from controllabilitytestforall import ControllabilityTestforAll
from buffer import Buffer
from env.simpleocp import SimpleOCP, SimpleOCPwoControl
from env.massspring import MassSpring, MassSpringwoControl
from env.pendulum import Pendulum, PendulumwoControl
from env.veh3dof import Veh3DoF, Veh3DoFwoControl
from env.oscillator import Oscillator, OscillatorwoControl
from env.lorenz import Lorenz, LorenzwoControl
from env.tunnel_diode import TunnelDiode
from env.carfollowing import CarFollowing, CarFollowingwoControl
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sample", type=int, default=5000, help="Number of samples")
    parser.add_argument("--env", type=str, default="TunnelDiode", help="env class name")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon value")
    # parser.add_argument("--target_state", type=float, nargs='+', default=[0, 0, 30], help="Target state")
    state = np.array([0.8844298, 0.210380361])
    # state = np.array([0.06263583, 0.75824183])
    parser.add_argument("--target_state", type=float, nargs='+', default=state, help="Target state")
    parser.add_argument("--lipschitz_confidence", type=float, default=0.2, help="Lipschitz confidence")
    parser.add_argument("--search_mode", type=str, default="max_radius", help="Search mode")
    parser.add_argument("--expand_mode", type=str, default="MECS", help="Expand mode")
    args = parser.parse_args()

    env = eval(args.env)(seed = 1)
    buffer = Buffer(buffer_size=args.num_sample)
    target_state = np.array(args.target_state)

    print(f"env: {args.env}")
    print(f"epsilon: {args.epsilon}")
    print(f"target_state: {args.target_state}")
    print(f"lipschitz_confidence: {args.lipschitz_confidence}")
    print(f"num_sample: {args.num_sample}")

    test = ControllabilityTest(
        env=env,
        buffer=buffer,
        target_state=target_state,
        epsilon=args.epsilon,
        num_sample=args.num_sample,
        lipschitz_confidence=args.lipschitz_confidence,
        use_kd_tree=True,
        lips_estimate_mode="sampling",
        expand_plot_interval=1000,
        backward_plot_interval=100000000000000,
        plot_expand_flag=False,
        plot_backward_flag=False,
        search_mode=args.search_mode,
        expand_mode=args.expand_mode,
    )

    test.run()
