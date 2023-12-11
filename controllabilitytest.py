# controllability test for dynamics systems in datative setting
from dataclasses import dataclass
import numpy as np
import gym

from buffer import Buffer
from typing import List, Optional

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
        self.onestep_matrix = np.zeros((2 * num_sample, 2 * num_sample))
        self.reachable_matrix = None
        

    def sample(self, num_sample: Optional[int] = None):
        state = self.env.reset()
        num_sample = self.num_sample if num_sample is None else num_sample
        for step in range(num_sample):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.buffer.add((state, action, reward, next_state, done))
            state = self.env.reset() if done else next_state

    def backward_expansion(self, neighbor: NeighbourSet) -> List(NeighbourSet):
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
                    ) / self.lipschitz_fx(neighbor.centered_state, transitions[0])
                )
            ) 
            for transitions in self.buffer.buffer[states_in_buffer_index]
        ]
        
    @staticmethod
    def distance(state1: np.ndarray, state2: np.ndarray) -> float:
        return np.linalg.norm(state1 - state2)
    
    def lipschitz_fx(self, state1: np.ndarray, state2: np.ndarray) -> float:
        # TODO: implement the lipschitz constant of the dynamics function
        return 1/2

    def clear(self):
        self.onestep_matrix = np.zeros((self.num_sample, self.num_sample))
        self.reachable_matrix = None
        self.buffer.clear()

    def seed(self, seed=None):
        self.env.seed(seed)