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
import numpy as np
import argparse
import matplotlib.pyplot as plt


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num_sample", type=int, default=5000, help="Number of samples")
#     parser.add_argument("--env", type=str, default="TunnelDiode", help="env class name")
#     parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon value")
#     # parser.add_argument("--target_state", type=float, nargs='+', default=[0, 0, 30], help="Target state")
#     parser.add_argument("--target_state", type=float, nargs='+', default=[0.8844298, 0.210380361], help="Target state")
#     parser.add_argument("--lipschitz_confidence", type=float, default=0.2, help="Lipschitz confidence")
#     parser.add_argument("--expand_mode", type=str, default="strict", help="Expand mode")
#     args = parser.parse_args()
#
#     env = eval(args.env)(seed = 1)
#     buffer = Buffer(buffer_size=args.num_sample)
#     target_state = np.array(args.target_state)
#
#     print(f"env: {args.env}")
#     print(f"epsilon: {args.epsilon}")
#     print(f"target_state: {args.target_state}")
#     print(f"lipschitz_confidence: {args.lipschitz_confidence}")
#     print(f"expand_mode: {args.expand_mode}")
#     print(f"num_sample: {args.num_sample}")
#
#     test = ControllabilityTest(
#         env=env,
#         buffer=buffer,
#         target_state=target_state,
#         epsilon=args.epsilon,
#         num_sample=args.num_sample,
#         lipschitz_confidence=args.lipschitz_confidence,
#         use_kd_tree=True,
#         # expand_mode=args.expand_mode,
#         lips_estimate_mode="sampling",
#         expand_plot_interval=1000,
#         backward_plot_interval=100000000000000,
#         plot_expand_flag=True,
#         plot_backward_flag=False,
#     )
#
#     test.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sample", type=int, default=5000, help="Number of samples")
    parser.add_argument("--env", type=str, default="TunnelDiode", help="env class name")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon value")
    # parser.add_argument("--target_state", type=float, nargs='+', default=[0, 0, 30], help="Target state")
    # parser.add_argument("--target_state", type=float, nargs='+', default=[0, 0], help="Target state")
    parser.add_argument("--lipschitz_confidence", type=float, default=0.2, help="Lipschitz confidence")
    parser.add_argument("--expand_mode", type=str, default="strict", help="Expand mode")
    args = parser.parse_args()

    env = eval(args.env)(seed = 1)
    buffer = Buffer(buffer_size=args.num_sample)

    # print(f"env: {args.env}")
    # print(f"epsilon: {args.epsilon}")
    # print(f"lipschitz_confidence: {args.lipschitz_confidence}")
    # print(f"expand_mode: {args.expand_mode}")
    # print(f"num_sample: {args.num_sample}")
    coordinates = []
    for x in range(-3, 14):
        for y in range(-3, 14):
            coordinates.append([x / 10, y / 10])
    for i in range(len(coordinates)):
        target_state = np.array(coordinates[i])
        print(f"target_state: {target_state}")
        test = ControllabilityTestforAll(
            env=env,
            buffer=buffer,
            target_state=target_state,
            epsilon=args.epsilon,
            num_sample=args.num_sample,
            lipschitz_confidence=args.lipschitz_confidence,
            use_kd_tree=True,
            # expand_mode=args.expand_mode,
            lips_estimate_mode="sampling",
            expand_plot_interval=1000,
            backward_plot_interval=100000000000000,
            plot_expand_flag=True,
            plot_backward_flag=False,
        )

        test.run()
    # coordinates = []
    # values = []
    # for x in range(-3, 14):
    #     for y in range(-3, 14):
    #         coordinates.append([x / 10, y / 10])
    # with open(f"figs/{env}/count.txt", "r") as file:
    #     for line in file:
    #         # 将读取的行转换为具体的值，并添加到values列表中
    #         value = float(line.strip())  # 假设文件中每行只包含一个数字，并且去除每行末尾的换行符
    #         values.append(value)
    # values = np.array(values).reshape(18, 18)
    # x = np.linspace(-1, 1, 18)
    # y = np.linspace(-1, 1, 18)
    # x, y = np.meshgrid(x, y)
    # z = np.sin(np.sqrt(x ** 2 + y ** 2))
    #
    # # 绘制三维图
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(x, y, values, cmap='YlOrBr')  # 这里使用'hot'颜色映射，你可以根据需要选择不同的颜色映射
    # fig.colorbar(surf)  # 添加颜色条
    # ax.set_xlabel("$y$", fontsize=14, fontname='Times New Roman')
    # ax.set_ylabel("$\dot{y}$", fontsize=14, fontname='Times New Roman')
    # ax.set_zlabel("$\epsilon$-controllable Ratio", fontsize=14, fontname='Times New Roman')
    #
    # plt.show()