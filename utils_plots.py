from typing import List
import os

import matplotlib.pyplot as plt

FILEPATH =  os.path.dirname(os.path.abspath(__file__))

class PlotUtils():
    def __init__(
            self, 
            obs_space, 
            action_space,
            orgin_state,
            orgin_radius,
        ):
        self.obs_space = obs_space
        self.action_space = action_space
        self.backward_counter = 0
        self.orgin_state = orgin_state
        self.orgin_radius = orgin_radius

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
        plt.savefig(os.path.join(FILEPATH, "./figs/epsilon_controllable_list.png"))
        plt.close()

    def plot_backward(self, state, r, next_state, next_r, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
            self.backward_counter = 0
            ax.set_xlim([self.obs_space.low[0], self.obs_space.high[0]])
            ax.set_ylim([self.obs_space.low[1], self.obs_space.high[1]])
            ax.axis('equal')
            # plot orgin state with radius orgin_radius
            circle = plt.Circle(self.orgin_state, self.orgin_radius, color='k', fill=False)
            ax.add_patch(circle)
        # plot circle at state with radius r
        circle = plt.Circle(state, r, color='r', fill=False)
        ax.add_patch(circle)
        # plot circle at next_state with radius next_r
        circle = plt.Circle(next_state, next_r, color='b', fill=False)
        ax.add_patch(circle)
        # plot line between state and next_state
        ax.plot([state[0], next_state[0]], [state[1], next_state[1]], color='g')
        ax.set_xlabel("state1")
        ax.set_ylabel("state2")
        # save figure
        self.backward_counter += 1
        plt.savefig(os.path.join(FILEPATH, "./figs/epsilon_controllable_list.png"))
        return fig, ax
