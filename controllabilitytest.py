# controllability test for dynamics systems in datative setting
from dataclasses import dataclass
import numpy as np
import gym

from buffer import Buffer
from typing import List, Optional, Tuple

@dataclass
class NeighbourSet:
    centered_state: np.ndarray
    radius: float
    visited: bool = False

class  ControllabilityTest:
    def __init__(self, env: gym.Env , buffer: Buffer, num_sample: int = 10000, epsilon: float = 0.05, lipschitz_confidence: float = np.inf):
        self.env = env
        self.buffer = buffer
        self.num_sample = num_sample
        self.epsilon = epsilon
        self.lipschitz_confidence = lipschitz_confidence
        self.epsilon_controllable_list: List[NeighbourSet] = []
        
    def sample(self, num_sample: Optional[int] = None):
        state = self.env.reset()
        num_sample = self.num_sample if num_sample is None else num_sample
        for step in range(num_sample):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.buffer.add((state, action, reward, next_state, done))
            state = self.env.reset() if done else next_state

    def backward_expand(self, neighbor: NeighbourSet) -> List[NeighbourSet]:
        '''
        :param neighbor (NeighbourSet): the neighbor set to be expanded
        :return (List[NeighbourSet]): the expanded neighbor set
        '''
        assert not neighbor.visited, "The neighbor set has been visited!"
        neighbor.visited = True
        # step1: finds all the states in the buffer that belong to the neighborhood set
        states_in_buffer_index = [
            idx for idx, transition in enumerate(self.buffer.buffer) 
            if self.distance(transition[3], neighbor.centered_state) <= neighbor.radius
        ]
        # step2: backward expand the states find in step1
        return [
            NeighbourSet(
                transitions[0], 
                min(
                    self.lipschitz_confidence, 
                    (
                        neighbor.radius - self.distance(
                            transitions[3], 
                            neighbor.centered_state
                        )
                    ) / self.lipschitz_fx(transitions[0])
                )
            )for transitions in [self.buffer.buffer[i] for i in states_in_buffer_index]
        ]
    
    def get_epsilon_controllable_set(self, state: np.ndarray):
        '''
        :param state (np.ndarray): the state to be tested
        :param epsilon (float): the epsilon value
        '''
        assert len(self.epsilon_controllable_list) == 0, "The epsilon controllable list is not empty!"
        self.epsilon_controllable_list.append(NeighbourSet(state, self.epsilon))
        # until all the neighbor sets are visited
        while not all([neighbor.visited for neighbor in self.epsilon_controllable_list]):
            for neighbor in self.epsilon_controllable_list:
                if not neighbor.visited:
                    expand_neighbor_list = self.backward_expand(neighbor)
                    for expand_neighbor in expand_neighbor_list:                    
                        is_repeat, idx = self.check_expand_neighbor_state_in_epsilon_controllable_list(expand_neighbor)
                        if is_repeat:
                            if expand_neighbor.radius > self.epsilon_controllable_list[idx].radius:
                                self.epsilon_controllable_list[idx].radius = expand_neighbor.radius
                        else:
                            self.epsilon_controllable_list.append(expand_neighbor)

    @staticmethod
    def distance(state1: np.ndarray, state2: np.ndarray) -> float:
        return np.linalg.norm(state1 - state2, ord=2)
    
    def check_expand_neighbor_state_in_epsilon_controllable_list(self, expand_neighbor: NeighbourSet) -> Tuple[bool, int]:
        idx_list = []
        for idx, neighbor in enumerate(self.epsilon_controllable_list):
            if self.distance(expand_neighbor.centered_state, neighbor.centered_state) < 1e-6:
                idx_list.append(idx)
        if len(idx_list) == 0:
            return False, -1
        elif len(idx_list) == 1:
            return True, idx_list[0]
        else:
            raise ValueError("The expand_neighbor is in the epsilon_controllable_list more than once!")

    def lipschitz_fx(self, state: np.ndarray) -> float:
        states_in_buffer_index = [
            idx for idx, transition in enumerate(self.buffer.buffer) 
            if self.distance(transition[0], state) <= self.lipschitz_confidence
        ]
        # TODO: implement the lipschitz constant of the dynamics function
        return 1/2

    def clear(self):
        self.epsilon_controllable_list = []
        self.buffer.clear()


    def seed(self, seed=None):
        self.env.seed(seed)