from typing import List
import numpy as np
import matplotlib.pyplot as plt

from controllabilitytest import NeighbourSet

def plot_epsilon_controllable_list(epsilon_controllable_list: List[NeighbourSet], save_path: str):
    plt.figure()
    for neighbor in epsilon_controllable_list:
        print("centered_state: {}, radius: {}".format(neighbor.centered_state, neighbor.radius))
        circle = plt.Circle(neighbor.centered_state, neighbor.radius, color='r', fill=False)
        plt.gca().add_patch(circle)
    plt.axis('equal')
    plt.xlabel("state")
    plt.ylabel("state")
    plt.savefig(save_path)
    plt.close()