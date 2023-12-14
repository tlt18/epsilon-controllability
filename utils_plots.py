from typing import List
import os

import matplotlib.pyplot as plt

FILEPATH =  os.path.dirname(os.path.abspath(__file__))

class PlotUtils():
    def __init__(
            self, 
            obs_space, 
            action_space,
            orgin_radius,
            date_time
        ):
        self.obs_space = obs_space
        self.action_space = action_space
        self.backward_counter = 0
        self.orgin_radius = orgin_radius
        self.data_time = date_time

    def set_orgin_state(self, orgin_state):
        self.orgin_state = orgin_state

    def plot_epsilon_controllable_list(self, epsilon_controllable_list: List):
        plt.figure()
        plt.xlim([self.obs_space.low[0], self.obs_space.high[0]])
        plt.ylim([self.obs_space.low[1], self.obs_space.high[1]])
        for neighbor in epsilon_controllable_list:
            print("centered_state: {}, radius: {}".format(neighbor.centered_state, neighbor.radius))
            circle = plt.Circle(neighbor.centered_state, neighbor.radius, color='r', fill=False)
            plt.gca().add_patch(circle)
        plt.axis('equal')
        plt.xlabel("state1")
        plt.ylabel("state2")
        plt.savefig(os.path.join(FILEPATH, f"figs/{self.data_time}/epsilon_controllable_list.png"))
        plt.close()

    def plot_backward(self, state, r, next_state, next_r, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
            self.backward_counter = 0
            ax.axis('equal')
            # plot orgin state with radius orgin_radius
            assert hasattr(self, "orgin_state"), "Please set orgin state first!"
            circle = plt.Circle(self.orgin_state, self.orgin_radius, color='k', fill=False)
            ax.add_patch(circle)
            # ax.set_xlim([self.obs_space.low[0], self.obs_space.high[0]])
            # ax.set_ylim([self.obs_space.low[1], self.obs_space.high[1]])
        # plot line between state and next_state
        ax.plot([state[0], next_state[0]], [state[1], next_state[1]], color='lightgray')
        # plot circle at state with radius r
        circle = plt.Circle(state, r, color='red', fill=False)
        ax.add_patch(circle)
        # plot circle at next_state with radius next_r
        circle = plt.Circle(next_state, next_r, color='cornflowerblue', fill=False)
        ax.add_patch(circle)
        ax.set_xlabel("state1")
        ax.set_ylabel("state2")

        # save figure
        self.backward_counter += 1
        # print("plot count: {}, expand state: {}, last state: {}".format(self.backward_counter, state, next_state))
        if self.backward_counter%1 == 0:
            plt.savefig(os.path.join(FILEPATH, f"figs/{self.data_time}/epsilon_controllable_list_{self.backward_counter}.png"))
        return fig, ax
    
    def plot_sample(self, sample_list: List):
        plt.figure()
        # plt.xlim([self.obs_space.low[0], self.obs_space.high[0]])
        # plt.ylim([self.obs_space.low[1], self.obs_space.high[1]])
        # plot line with arrow
        for k in range(len(sample_list)):
            plt.plot([sample_list[k][0][0], sample_list[k][3][0]], [sample_list[k][0][1], sample_list[k][3][1]], color='lightgray', linewidth=0.5)
            plt.plot(sample_list[k][0][0], sample_list[k][0][1], 'o', color='cornflowerblue', markersize=2)
            plt.plot(sample_list[k][3][0], sample_list[k][3][1], 'o', color='cornflowerblue', markersize=2)
            plt.arrow(
                sample_list[k][0][0], 
                sample_list[k][0][1], 
                sample_list[k][3][0] - sample_list[k][0][0], 
                sample_list[k][3][1] - sample_list[k][0][1], 
                color='red',
                width = 0.001,
                head_width = 0.01,
            )
        plt.axis('equal')
        plt.xlabel("state1")
        plt.ylabel("state2")
        plt.savefig(os.path.join(FILEPATH, f"figs/{self.data_time}/sample.png"))
        plt.close()
