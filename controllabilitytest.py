# controllability test for dynamics systems in datative setting
from dataclasses import dataclass
import time
from typing import List, Optional, Tuple

import gym
import numpy as np

from buffer import Buffer
from utils_plots import PlotUtils

@dataclass
class NeighbourSet:
    centered_state: np.ndarray
    radius: float
    visited: bool = False

class  ControllabilityTest:
    def __init__(self, env: gym.Env , buffer: Buffer, num_sample: int = 10000, epsilon: float = 0.05, lipschitz_confidence: float = 1):
        self.env = env
        self.buffer = buffer
        self.num_sample = num_sample
        self.epsilon = epsilon
        self.lipschitz_confidence = lipschitz_confidence
        self.epsilon_controllable_list: List[NeighbourSet] = []
        
    def sample(self):
        state = self.env.reset()
        for step in range(self.num_sample):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.buffer.add((state, action, reward, next_state, done))
            state = self.env.reset() if done else next_state

    def backward_expand(self, neighbor: NeighbourSet) -> Tuple[List[NeighbourSet], List[NeighbourSet]]:
        '''
        :param neighbor (NeighbourSet): the neighbor set to be expanded
        :return (Tuple[List[NeighbourSet], List[NeighbourSet]]): the expanded neighbor set and last neighbor set
        '''
        assert not neighbor.visited, "The neighbor set has been visited!"
        neighbor.visited = True
        
        # Step 1: Find all the states in the buffer that belong to the neighborhood set
        states_in_buffer_index = [
            idx for idx, transition in enumerate(self.buffer.buffer) 
            if self.distance(transition[3], neighbor.centered_state) <= neighbor.radius
        ]
        
        # Step 2: Backward expand the states found in Step 1
        buffer_transitions = [self.buffer.buffer[i] for i in states_in_buffer_index]
        if len(buffer_transitions) == 0:
            return [], []
        else:
            neighbour_sets = [
                (
                    NeighbourSet(
                        transitions[0], 
                        min(
                            self.lipschitz_confidence, 
                            (neighbor.radius - self.distance(transitions[3], neighbor.centered_state)) / self.lipschitz_fx(transitions[0])
                        )
                    ),
                    NeighbourSet(
                        transitions[3],
                        neighbor.radius - self.distance(transitions[3], neighbor.centered_state)
                    )
                )
                for transitions in buffer_transitions
            ]
            return tuple(map(list, zip(*neighbour_sets)))
    
    def get_epsilon_controllable_set(self, state: np.ndarray):
        '''
        :param state (np.ndarray): the state to be tested
        '''
        assert len(self.epsilon_controllable_list) == 0, "The epsilon controllable list is not empty!"
        count = 0
        fig, ax = None, None
        self.plot_utils = PlotUtils(
            obs_space = self.env.observation_space, 
            action_space = self.env.action_space,
            orgin_state = state, 
            orgin_radius = self.epsilon,
        )

        self.epsilon_controllable_list.append(NeighbourSet(state, self.epsilon))
        # until all the neighbor sets are visited
        while not all([neighbor.visited for neighbor in self.epsilon_controllable_list]):
            for neighbor in self.epsilon_controllable_list:
                # TODO: more detailed implementation  
                if len(self.epsilon_controllable_list) == self.num_sample:
                    return
                if not neighbor.visited:
                    expand_neighbor_list, last_neighbor_list = self.backward_expand(neighbor)
                    for idx_expland, expand_neighbor in enumerate(expand_neighbor_list):
                        relation, idx_inlist = self.check_expand_neighbor_relation(expand_neighbor)
                        if relation == None:
                            self.epsilon_controllable_list.append(expand_neighbor)
                        elif relation == "list_in_expand":
                            # del self.epsilon_controllable_list[i] for i in idx_list
                            for i in reversed(idx_inlist): # there idx_inlist is sorted in ascending order
                                del self.epsilon_controllable_list[i]
                            self.epsilon_controllable_list.append(expand_neighbor)
                        else:
                            assert relation == "expand_in_list", "relation is not correct!"
                        if relation != "expand_in_list":
                            fig, ax = self.plot_utils.plot_backward(
                                expand_neighbor.centered_state, 
                                expand_neighbor.radius, 
                                last_neighbor_list[idx_expland].centered_state, 
                                last_neighbor_list[idx_expland].radius,
                                fig=fig,
                                ax=ax
                            )
                    count += 1
                    print("expand count: {}, new_neighbor_num: {}, total_controllable_num: {}"
                        .format(count, len(expand_neighbor_list), len(self.epsilon_controllable_list))
                    )

    def run(self, state: np.ndarray):
        time_start = time.time()
        self.sample()
        time_sample = time.time() - time_start
        print("time for sampling: {:.4f}s".format(time_sample))

        self.plot_utils.plot_sample(self.buffer.buffer)
        time_plot = time.time() - time_start - time_sample
        print("time for plotting: {:.4f}s".format(time_plot))

        self.get_epsilon_controllable_set(state)
        time_calonestep = time.time() - time_start - time_sample - time_plot
        print("time for calculating epsilon controllable set: {:.4f}s".format(time_calonestep))

        self.plot_utils.plot_sample(self.buffer.buffer)

    @staticmethod
    def distance(state1: np.ndarray, state2: np.ndarray) -> float:
        return np.linalg.norm(state1 - state2, ord=2)
    
    def check_expand_neighbor_relation(self, expand_neighbor: NeighbourSet) -> Tuple[Optional[str], List[int]]:
        idx_list = []
        for idx, neighbor in enumerate(self.epsilon_controllable_list):
            dist = self.distance(expand_neighbor.centered_state, neighbor.centered_state)
            if dist <= neighbor.radius - expand_neighbor.radius:
                return "expand_in_list", [idx]
            elif dist <= expand_neighbor.radius - neighbor.radius:
                idx_list.append(idx)
        if len(idx_list) > 0:
            return "list_in_expand", idx_list
        else:
            return None, [-1]

    def lipschitz_fx(self, state: np.ndarray) -> float:
        states_in_buffer_index = [
            idx for idx, transition in enumerate(self.buffer.buffer) 
            if self.distance(transition[0], state) <= self.lipschitz_confidence
        ]
        # TODO: implement the lipschitz constant of the dynamics function
        return 1.2

    def clear(self):
        self.epsilon_controllable_list = []
        self.buffer.clear()

    def seed(self, seed=None):
        self.env.seed(seed)