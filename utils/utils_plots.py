from typing import List
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
FILEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PlotUtils():
    def __init__(
            self, 
            obs_space, 
            action_space,
            orgin_state,
            orgin_radius,
            fig_title,
            backward_plot_interval=100,
        ):
        self.obs_space = obs_space
        self.action_space = action_space
        self.backward_counter = 0
        self.orgin_state = orgin_state
        self.orgin_radius = orgin_radius
        self.fig_title = fig_title
        self.backward_plot_interval = backward_plot_interval
        if obs_space.shape[0] == 2:
            self.plot_epsilon_controllable_set = self.plot_epsilon_controllable_set_2D
            self.plot_backward = self.plot_backward_2D
            self.plot_sample = self.plot_sample_2D
            self.plot_controllable_data = self.plot_controllable_data_2D
        elif obs_space.shape[0] == 3:
            self.plot_epsilon_controllable_set = self.plot_epsilon_controllable_set_3D
            self.plot_backward = self.plot_backward_3D
            self.plot_sample = self.plot_sample_3D
            self.plot_controllable_data = self.plot_controllable_data_3D
        else:
            self.plot_epsilon_controllable_set = self.empty_function
            self.plot_backward = self.empty_function
            self.plot_sample = self.empty_function
            print("Warning: obs_space.shape[0] is not 2 or 3, plot_epsilon_controllable_set, plot_backward and plot_sample are not defined.")

    def plot_epsilon_controllable_set_2D(self, epsilon_controllable_list, expand_counter: int, transitions,target_state):
        plt.figure()
        # plt.scatter(transitions.state[:, 0], transitions.state[:, 1], marker='o', color='cornflowerblue', s=1)

        for k in range(len(transitions)):
            # plt.plot.py([sample_list[k][0][0], sample_list[k][3][0]], [sample_list[k][0][1], sample_list[k][3][1]], color='lightgray', linewidth=0.5)
            plt.arrow(
                transitions[k].state[0],
                transitions[k].state[1],
                transitions[k].next_state[0] - transitions[k].state[0],
                transitions[k].next_state[1] - transitions[k].state[1],
                color=(1, 204 / 255, 153 / 255),
                width=0.002,
                head_width=0.02,
            )
        plt.arrow(
            transitions[0].state[0],
            transitions[0].state[1],
            transitions[0].next_state[0] - transitions[0].state[0],
            transitions[0].next_state[1] - transitions[0].state[1],
            color=(1, 204 / 255, 153 / 255),
            width=0.002,
            head_width=0.02,
        )
        plt.scatter(transitions.state[:, 0], transitions.state[:, 1], marker='o', color=(247/255,86/255,59/255), s=1)
        plt.scatter(transitions.next_state[:, 0], transitions.next_state[:, 1], marker='o', color=(247/255,86/255,59/255),label="state point", s=1)

        for neighbor in epsilon_controllable_list:
            circle = plt.Circle(neighbor.centered_state, neighbor.radius, color='cornflowerblue', fill=True, alpha=0.2)
            plt.gca().add_patch(circle)
        # circle2 = plt.Circle((target_state[0], target_state[1]), 0.05, color='cornflowerblue', fill=True, alpha=0.2)
        # plt.add_patch(circle2)
        # plt.plot(target_state[0]+0.05, target_state[1]+0.05, marker='o', markersize=10, color='cornflowerblue', label='epsilon controllable set')
        plt.scatter(target_state[0], target_state[1], color='yellow', label='target state', marker='*')
        #
        # plt.xlim([-0.5, 1.5])
        #
        # plt.ylim([-0.4, 1.2])
        # plt.xlim([-2, 2.0])
        # plt.ylim([-4.0, 4.0])
        plt.xlabel("$x_1$",fontsize=14,fontname= 'Times New Roman')
        plt.ylabel("$x_2$",fontsize=14,fontname= 'Times New Roman')
        # plt.title(f"expand_{expand_counter}")
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(FILEPATH, f"figs/{self.fig_title}/epsilon_controllable_set/{expand_counter}.pdf"))
        plt.savefig(os.path.join(FILEPATH, f"figs/{self.fig_title}/epsilon_controllable_set/{expand_counter}.pdf"))
        plt.close()

    def plot_epsilon_controllable_set_3D(self, epsilon_controllable_list, expand_counter: int,transitions,target_state):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # for k in range(len(transitions)):
        #     ax.quiver(
        #         transitions[k].state[0],
        #         transitions[k].state[1],
        #         transitions[k].state[2],
        #         transitions[k].next_state[0] - transitions[k].state[0],
        #         transitions[k].next_state[1] - transitions[k].state[1],
        #         transitions[k].next_state[2] - transitions[k].state[2],
        #         color=(1, 204/255, 153/255),
        #         length=0.1,
        #         normalize=True,
        #     )
        # ax.quiver(
        #     transitions[k].state[0],
        #     transitions[k].state[1],
        #     transitions[k].state[2],
        #     transitions[k].next_state[0] - transitions[k].state[0],
        #     transitions[k].next_state[1] - transitions[k].state[1],
        #     transitions[k].next_state[2] - transitions[k].state[2],
        #     color=(1, 204/255, 153/255),
        #     label='state transfer function',
        #     length=0.1,
        #     normalize=True,
        # )
        # ax.scatter(transitions.state[:, 0], transitions.state[:, 1], transitions.state[:, 2], marker='o', color='red', s=1)
        # ax.scatter(transitions.next_state[:, 0], transitions.next_state[:, 1], transitions.next_state[:, 2], marker='o', color='red', label='state point', s=0.8)
        ax.scatter(target_state[0], target_state[1], target_state[2], marker='o', color='yellow', label='target state', s=1)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        for neighbor in epsilon_controllable_list:
            x = neighbor.centered_state[0] + neighbor.radius * np.outer(np.cos(u), np.sin(v))
            y = neighbor.centered_state[1] + neighbor.radius * np.outer(np.sin(u), np.sin(v))
            z = neighbor.centered_state[2] + neighbor.radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='dodgerblue', alpha=0.4)
            ax.set_box_aspect([1,1,1])
        ax.set_xlabel('state-1')
        ax.set_ylabel('state-2')
        ax.set_zlabel('state-3')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plt.legend()
        plt.savefig(os.path.join(FILEPATH, f"figs/{self.fig_title}/epsilon_controllable_set/{expand_counter}.png"))
        plt.close()

    def plot_backward_2D(self, state, r, next_state, next_r, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
            self.backward_counter = 0
            ax.axis('equal')
            # plot.py orgin state with radius orgin_radius
            assert hasattr(self, "orgin_state"), "Please set orgin state first!"
            circle = plt.Circle(self.orgin_state, self.orgin_radius, color='k', fill=False)
            ax.add_patch(circle)
            # ax.set_xlim([self.obs_space.low[0], self.obs_space.high[0]])
            # ax.set_ylim([self.obs_space.low[1], self.obs_space.high[1]])
        # plot.py line between state and next_state
        ax.plot([state[0], next_state[0]], [state[1], next_state[1]], color='lightgray', alpha=0.5, linewidth=0.5)
        # plot.py circle at state with radius r
        circle = plt.Circle(state, r, color='red', fill=False)
        ax.add_patch(circle)
        # plot.py circle at next_state with radius next_r
        circle = plt.Circle(next_state, next_r, color='cornflowerblue', fill=False)
        ax.add_patch(circle)
        ax.set_xlabel("state1")
        ax.set_ylabel("state2")
        ax.set_title(f"backward_{self.backward_counter}")
        # save figure
        if self.backward_counter%self.backward_plot_interval == 0:
            plt.savefig(os.path.join(FILEPATH, f"figs/{self.fig_title}/expand_backward/{self.backward_counter}.png"))
        self.backward_counter += 1
        return fig, ax
    
    def plot_backward_3D(self, state, r, next_state, next_r, fig=None, ax=None):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            self.backward_counter = 0
            # plot.py orgin state with radius orgin_radius
            assert hasattr(self, "orgin_state"), "Please set orgin state first!"
            x = self.orgin_state[0] + self.orgin_radius * np.outer(np.cos(u), np.sin(v))
            y = self.orgin_state[1] + self.orgin_radius * np.outer(np.sin(u), np.sin(v))
            z = self.orgin_state[2] + self.orgin_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='k', alpha=0.1)
        # plot.py line between state and next_state
        ax.plot([state[0], next_state[0]], [state[1], next_state[1]], [state[2], next_state[2]], color='lightgray', alpha=0.6, linewidth=0.5)
        # plot.py circle at state with radius r
        x = state[0] + r * np.outer(np.cos(u), np.sin(v))
        y = state[1] + r * np.outer(np.sin(u), np.sin(v))
        z = state[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='red', alpha=0.4)
        # plot.py circle at next_state with radius next_r
        x = next_state[0] + next_r * np.outer(np.cos(u), np.sin(v))
        y = next_state[1] + next_r * np.outer(np.sin(u), np.sin(v))
        z = next_state[2] + next_r * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='cornflowerblue', alpha=0.4)
        ax.set_box_aspect([1,1,1])
        ax.set_xlabel('state1')
        ax.set_ylabel('state2')
        ax.set_zlabel('state3')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.set_title(f"backward_{self.backward_counter}")
        # save figure
        if self.backward_counter%self.backward_plot_interval == 0:
            plt.savefig(os.path.join(FILEPATH, f"figs/{self.fig_title}/expand_backward/{self.backward_counter}.png"))
        self.backward_counter += 1
        return fig, ax
    
    def plot_sample_2D(self, transitions):
        plt.figure()
        # plt.axis([-2, 2, -8, 8])
        # plt.xlim([-2.0, 2.0])
        # plt.ylim([-4.5, 4.5])
        # plt.xlim([self.obs_space.low[0], self.obs_space.high[0]])
        # plt.ylim([self.obs_space.low[1], self.obs_space.high[1]])
        # plot.py line with arrow

        for k in range(len(transitions)):
            # plt.plot.py([sample_list[k][0][0], sample_list[k][3][0]], [sample_list[k][0][1], sample_list[k][3][1]], color='lightgray', linewidth=0.5)
            plt.arrow(
                transitions[k].state[0],
                transitions[k].state[1],
                transitions[k].next_state[0] - transitions[k].state[0],
                transitions[k].next_state[1] - transitions[k].state[1],
                color=(1, 204/255, 153/255),
                width = 0.001,
                head_width = 0.02,
            )
        plt.scatter(transitions.state[:, 0], transitions.state[:, 1], marker='o', color=(247/255,86/255,59/255), s=1)
        plt.scatter(transitions.next_state[:, 0], transitions.next_state[:, 1], marker='o', color=(247/255,86/255,59/255), label='state point', s=0.8)
        # plt.scatter(0.8844298, 0.210380361, marker='*', color='blue', label='equilibrium points',
        #            s=10)
        # plt.scatter(0.06263583,0.75824183, marker='*', color='blue',
        #            s=10)
        # plt.axis('equal')
        plt.xlabel("$y$",fontsize=14,fontname= 'Times New Roman')
        plt.ylabel("$\dot{y}$",fontsize=14,fontname= 'Times New Roman')
        # plt.xlabel("$x_1$",fontsize=14,fontname= 'Times New Roman')
        # plt.ylabel("$x_2$",fontsize=14,fontname= 'Times New Roman')
        # plt.xlim([-3.0, 3.0])
        # x_lim = plt.xlim([-2.0, 2.0])
        # print("当前 x 轴范围为:", x_lim)
        plt.legend(fontsize=12)
        plt.savefig(os.path.join(FILEPATH, f"figs/{self.fig_title}/sample.pdf"))
        plt.close()


    def plot_sample_3D(self, transitions):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot.py line with arrow
        for k in range(len(transitions)):
            ax.quiver(
                transitions[k].state[0], 
                transitions[k].state[1], 
                transitions[k].state[2], 
                transitions[k].next_state[0] - transitions[k].state[0], 
                transitions[k].next_state[1] - transitions[k].state[1], 
                transitions[k].next_state[2] - transitions[k].state[2], 
                color=(1, 204/255, 153/255),
                length=0.1,
                normalize=True,
            )
        ax.quiver(
            transitions[k].state[0],
            transitions[k].state[1],
            transitions[k].state[2],
            transitions[k].next_state[0] - transitions[k].state[0],
            transitions[k].next_state[1] - transitions[k].state[1],
            transitions[k].next_state[2] - transitions[k].state[2],
            color=(1, 204/255, 153/255),
            label='state transfer function',
            length=0.1,
            normalize=True,
        )
        ax.scatter(transitions.state[:, 0], transitions.state[:, 1], transitions.state[:, 2], marker='o', color='red', s=1)
        ax.scatter(transitions.next_state[:, 0], transitions.next_state[:, 1], transitions.next_state[:, 2], marker='o', color='red', label='state point', s=0.8)

        # ax.set_box_aspect([1,1,1])
        ax.set_xlabel('state-1')
        ax.set_ylabel('state-2')
        ax.set_zlabel('state-3')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plt.savefig(os.path.join(FILEPATH, f"figs/{self.fig_title}/sample.pdf"))
        plt.close()

    def plot_controllable_data_2D(self, dataset):
        controllable_data = dataset[dataset.is_controllable == True]
        plt.figure()
        plt.scatter(controllable_data.state[:, 0], controllable_data.state[:, 1], marker='o', color='cornflowerblue', s=1)
        plt.axis('equal')
        plt.xlim([self.obs_space.low[0], self.obs_space.high[0]])
        plt.ylim([self.obs_space.low[1], self.obs_space.high[1]])
        plt.xlabel("state1")
        plt.ylabel("state2")
        plt.savefig(os.path.join(FILEPATH, f"figs/{self.fig_title}/controllable_data.png"))
        plt.close()

    def plot_controllable_data_3D(self, dataset):
        controllable_data = dataset[dataset.is_controllable == True]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(controllable_data.state[:, 0], controllable_data.state[:, 1], controllable_data.state[:, 2], marker='o', color='cornflowerblue', s=1)
        # ax.set_box_aspect([1,1,1])
        ax.set_xlabel('state1')
        ax.set_ylabel('state2')
        ax.set_zlabel('state3')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        plt.savefig(os.path.join(FILEPATH, f"figs/{self.fig_title}/controllable_data.png"))
        plt.close()

    def save_figs(self, fig, ax):
        ax.set_title(f"backward_{self.backward_counter}")
        plt.savefig(os.path.join(FILEPATH, f"figs/{self.fig_title}/expand_backward/{self.backward_counter}.png"))

    def empty_function(*args, **kwargs):
        return None, None