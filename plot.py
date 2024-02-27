import matplotlib.pyplot as plt
import numpy as np

plt.figure()

epsilon_list = np.arange(0.01, 0.11, 0.01)
ratio_list = np.full(len(epsilon_list), 0.9996)
ratio_list2 = np.array([0.0018,0.1402,0.8576,0.9876,0.9984,0.9996,0.9998,1.0,1,1])
plt.plot(epsilon_list, ratio_list2, ls='-', color=(1, 204/255, 153/255), marker='^',label='target state=[0.0,0.0]')
# plt.plot(epsilon_list, ratio_list2, ls='--', color='red', marker='s',label='target state=[-0.25,0.0]')
        # # plt.scatter(transitions.state[:, 0], transitions.state[:, 1], marker='o', color='cornflowerblue', s=1)
        # plt.scatter(transitions.state[:, 0], transitions.state[:, 1], marker='o', color='red', s=1)
        # plt.scatter(transitions.next_state[:, 0], transitions.next_state[:, 1], marker='o', color='red', s=1)
        #
        # for neighbor in epsilon_controllable_list:
        #     circle = plt.Circle(neighbor.centered_state, neighbor.radius, color='cornflowerblue', fill=True, alpha=0.2)
        #     plt.gca().add_patch(circle)
        # # circle2 = plt.Circle((0.25, 0.0), 0.05, color='orange', label='original epsilon controllable set', fill=True, alpha=0.2)
        # # plt.gca().add_patch(circle2)
        # plt.scatter(target_state[0], target_state[1], color='yellow', label='target state', marker='*')
        # plt.axis('equal')
        # plt.xlim([self.obs_space.low[0], self.obs_space.high[0]])
        # plt.ylim([self.obs_space.low[1], self.obs_space.high[1]])
plt.xlabel("Size of epsilon")
plt.ylabel("Ratio between controllable points and sample points.")
plt.title("Ratio between controllable points and sample points in massspring")
plt.legend()
plt.show()
        # # plt.savefig(os.path.join(FILEPATH, f"figs/{self.fig_title}/epsilon_controllable_set/{expand_counter}.pdf"))
        # # plt.savefig(os.path.join(FILEPATH, f"figs/{self.fig_title}/epsilon_controllable_set/{expand_counter}.pdf"))
        # plt.close()