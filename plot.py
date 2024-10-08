import matplotlib.pyplot as plt
import numpy as np

plt.figure()

epsilon_list = np.arange(0.01, 0.21, 0.01)
ratio_list = np.full(len(epsilon_list), 0.9996)
ratio_list2 = np.array([0.0888,0.2998,0.5998,0.8998,1,1,1,1.0,1,1])
# ratio_list3 = np.array([0.00,0.1808,0.4996,0.8996,0.9996,0.9996,0.9998,1,1,1])
# # ratio_list4 = np.array([0.0918,0.2,0.3,0.4,0.6,0.72,0.94,1,1,1])
# #
# ratio_list2 = np.array([0.0,0,0,0.006,0.024,0.048,0.078,0.204,0.298,0.436])
# ratio_list3 = np.array([0.026,0.0694,0.1088,0.1492,0.1906,0.203,0.23,0.274,0.301,0.3218])
ratio_list4 = np.array([0.026,0.0478,0.0678,0.1274,0.151,0.1808,0.2176,0.2596,0.2844,0.3236])
#
# ratio_list2 = np.array([0.8,1.0,1.0,1,1,1,1,1.0,1,1])
# ratio_list3 = np.array([0.00,0.0302,0.0862,0.1084,0.1448,0.1466,0.1476,0.1776,0.2298,0.233])
# ratio_list4 = np.array([0.0,0.0268,0.0524,0.084,0.1044,0.1258,0.141,0.1578,0.163,0.1646])
# plt.plot(epsilon_list, ratio_list2, ls='-', color=(1, 204/255, 153/255), marker='^',label='target state=[0.0,0.0]')
# ratio_list2 = np.array([0,0.0583,0.181,0.3086,0.443,0.512,0.582,0.661,0.75,0.778,
#                         0.836,0.895,0.909,0.927,0.950,0.954,0.956,0.97,0.971,0.973])
# ratio_list3 = np.array([0,0.0,0,0,0,0,0,0,0,0,
#                         0,0,0.016,0.099,0.227,0.381,0.477,0.53,0.624,0.721])
# ratio_list4 = np.array([0,0.0,0,0,0,0,0,0,0,0,
#                         0,0,0.027,0.087,0.194,0.335,0.432,0.56,0.67,0.738])
# ratio_list5 = np.array([0,0.0,0,0,0,0,0,0,0,0.03,
#                         0.1,0.257,0.375,0.505,0.605,0.713,0.769,0.862,0.888,0.899])
#
# plt.plot(epsilon_list, ratio_list2, ls='-', color='red', marker='o', label='with control')
# plt.plot(epsilon_list, ratio_list3, ls='--', color='cornflowerblue', marker='s',label='target state=[4.5,0.25,0.0]')
# plt.plot(epsilon_list, ratio_list4, ls='-.', color='orange', marker='^',label='without control')
# plt.plot(epsilon_list, ratio_list5, ls='-', color='green', marker='*',label='target state=[4.5,0.25,0.25]')
#
# # plt.plot(epsilon_list, ratio_list2, ls='--', color='red', marker='s',label='target state=[-0.25,0.0]')
#         # # plt.scatter(transitions.state[:, 0], transitions.state[:, 1], marker='o', color='cornflowerblue', s=1)
#         # plt.scatter(transitions.state[:, 0], transitions.state[:, 1], marker='o', color='red', s=1)
#         # plt.scatter(transitions.next_state[:, 0], transitions.next_state[:, 1], marker='o', color='red', s=1)
#         #
#         # for neighbor in epsilon_controllable_list:
#         #     circle = plt.Circle(neighbor.centered_state, neighbor.radius, color='cornflowerblue', fill=True, alpha=0.2)
#         #     plt.gca().add_patch(circle)
#         # # circle2 = plt.Circle((0.25, 0.0), 0.05, color='orange', label='original epsilon controllable set', fill=True, alpha=0.2)
#         # # plt.gca().add_patch(circle2)
#         # plt.scatter(target_state[0], target_state[1], color='yellow', label='target state', marker='*')
#         # plt.axis('equal')
#         # plt.xlim([self.obs_space.low[0], self.obs_space.high[0]])
#         # plt.ylim([self.obs_space.low[1], self.obs_space.high[1]])
# plt.xlabel("Size of epsilon",fontsize=14,fontname= 'Times New Roman')
# plt.ylabel("$\epsilon$-controllable Ratio",fontsize=14,fontname= 'Times New Roman')
# # plt.title("Ratio between controllable points and sample points in massspring")
# plt.legend()
# plt.show()
if __name__ == "__main__":
    coordinates = []
    values = []
    a = np.array([0.9,0.3])
    for x in range(-10, 10):
        for y in range(-10, 10):
            coordinates.append([x / 10, y / 10])
    with open("figs/MassSpringwoControl/countall.txt", "r") as file:
        for line in file:
            # 将读取的行转换为具体的值，并添加到values列表中
            words = line.split()
            # 如果该行至少有两个单词（假设每行有两个数字）
            # if len(words) >= 2:
            #     # 提取第二个单词（假设第二个单词是要提取的数字）
            #     value = float(words[1])  # 假设第二个数字为要提取的数字
            values.append(float(words[1]))
    values = np.array(values).reshape(20,20)
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    x, y = np.meshgrid(x, y)
    z = np.sin(np.sqrt(x**2 + y**2))
    # plt.figure()
    # plt.plot(epsilon_list, values[0:20], ls='-', color='red', marker='o', label='$x_t=[0.8844298, 0.210380361]$')
    # plt.plot(epsilon_list, values[20:40], ls='--', color='cornflowerblue', marker='s',label='$x_t=[0.06263583, 0.75824183]$')
    # plt.plot(epsilon_list, values[40:60], ls='-.', color='orange', marker='^',label='$x_t=[0.285, 0.61]$')
    # plt.xlabel("Size of epsilon",fontsize=14,fontname= 'Times New Roman')
    # plt.ylabel("$\epsilon$-controllable Ratio",fontsize=14,fontname= 'Times New Roman')
    # plt.legend()
    # plt.show()
    # 绘制三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, values, cmap='YlOrBr')  # 这里使用'hot'颜色映射，你可以根据需要选择不同的颜色映射
    fig.colorbar(surf)  # 添加颜色条
    ax.set_xlabel("$y$", fontsize=14, fontname='Times New Roman')
    ax.set_ylabel("$\dot{y}$", fontsize=14, fontname='Times New Roman')
    ax.set_zlabel("$\epsilon$-controllable Ratio",fontsize=14,fontname= 'Times New Roman')
    print(a == np.array([0.9,0.3]))
    plt.show()
    # print(coordinates)
    #         plt.savefig(os.path.join(FILEPATH, f"figs/{self.fig_title}/epsilon_controllable_set/{expand_counter}.pdf"))
    #         plt.close()