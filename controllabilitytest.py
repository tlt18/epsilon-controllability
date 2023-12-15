# controllability test for dynamics systems in datative setting
from copy import deepcopy
from dataclasses import dataclass
import datetime
import os
import time
from typing import List, Optional, Tuple, Union

from cvxopt import solvers, matrix
import gym
import numpy as np
from sklearn.neighbors import KDTree

from buffer import Buffer
from utils.utils_plots import PlotUtils, FILEPATH
from utils.timeit import timeit, Timeit


@dataclass
class NeighbourSet:
    centered_state: np.ndarray
    radius: Union[float, np.ndarray]
    visited: Union[bool, np.ndarray] = False

    @staticmethod
    def batch(neighbor_list: List["NeighbourSet"]) -> 'NeighbourSet':
        return NeighbourSet(
            centered_state = np.array([neighbor.centered_state for neighbor in neighbor_list]),
            radius = np.array([neighbor.radius for neighbor in neighbor_list]),
            visited = np.array([neighbor.visited for neighbor in neighbor_list]),
        )
    
    def __len__(self):
        assert len(self.radius.shape) > 0, "The data must be batched!"
        assert len(self.centered_state) == len(self.radius) == len(self.visited), \
            "The length of centered_state, radius and visited are not equal!"
        return len(self.centered_state)
    
    def __getitem__(self, index):
        return NeighbourSet(
            centered_state = self.centered_state[index],
            radius = self.radius[index],
            visited = self.visited[index],
        )
    
    def __setitem__(self, index, neighbor: "NeighbourSet"):
        self.centered_state[index] = neighbor.centered_state
        self.radius[index] = neighbor.radius
        self.visited[index] = neighbor.visited
    
    def __delitem__(self, index):
        self.centered_state = np.delete(self.centered_state, index, axis=0)
        self.radius = np.delete(self.radius, index, axis=0)
        self.visited = np.delete(self.visited, index, axis=0)

    def append(self, neighbor: "NeighbourSet"):
        self.centered_state = np.append(self.centered_state, neighbor.centered_state.reshape(1, -1), axis=0)
        self.radius = np.append(self.radius, neighbor.radius.reshape(1), axis=0)
        self.visited = np.append(self.visited, neighbor.visited.reshape(1), axis=0)
        
class Transition:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray

    def __init__(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        self.state = state
        self.action = action
        self.next_state = next_state

    def __len__(self):
        return len(self.state)
    
    def __getitem__(self, index):
        return Transition(
            state = self.state[index],
            action = self.action[index],
            next_state = self.next_state[index],
        )


class  ControllabilityTest:
    def __init__(
            self, 
            env: gym.Env , 
            buffer: Buffer, 
            num_sample: int = 10000, 
            epsilon: float = 0.05, 
            lipschitz_confidence: float = 1,
            use_kd_tree: bool = False,
            expand_plot_interval: int = 1, 
            backward_plot_interval: int = 100,
            plot_flag: bool = False
        ):
        self.env = env
        self.buffer = buffer
        self.num_sample = num_sample
        self.epsilon = epsilon
        self.lipschitz_confidence = lipschitz_confidence
        self.use_kd_tree = use_kd_tree
        self.expand_plot_interval = expand_plot_interval
        self.plot_flag = plot_flag
        self.epsilon_controllable_set: NeighbourSet = None

        fig_title = f"{num_sample}samples-{epsilon}epsilon-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(FILEPATH + f"/figs/{fig_title}/epsilon_controllable_set", exist_ok=True)
        os.makedirs(FILEPATH + f"/figs/{fig_title}/expand_backward", exist_ok=True)
        self.plot_utils: PlotUtils = PlotUtils(
            obs_space = self.env.observation_space, 
            action_space = self.env.action_space,
            orgin_radius = self.epsilon,
            fig_title = fig_title,
            backward_plot_interval = backward_plot_interval,
        )
        self.dataset = None
        self.state_kdtree: KDTree = None
        self.next_state_kdtree: KDTree = None
        
    def sample(self):
        state = self.env.reset()
        for _ in range(self.num_sample):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.buffer.add((state, action, reward, next_state, done))
            state = self.env.reset() if done else next_state
        self.dataset = Transition(*self.buffer.get_data())
        if self.use_kd_tree:
            self.state_kdtree = KDTree(self.dataset.state, leaf_size=40, metric='euclidean')
            self.next_state_kdtree = KDTree(self.dataset.next_state, leaf_size=40, metric='euclidean')

    def backward_expand(self, neighbor: NeighbourSet) -> Tuple[NeighbourSet, NeighbourSet]:
        '''
        :param neighbor (NeighbourSet): the neighbor set to be expanded
        :return (Tuple[List[NeighbourSet], List[NeighbourSet]]): the expanded neighbor set and last neighbor set
        '''
        assert not neighbor.visited, "The neighbor set has been visited!"
        neighbor.visited = True
        
        # Step 1: Find all the states in the buffer that belong to the neighborhood set
        if self.use_kd_tree:
            data_in_neighbourhood = deepcopy(self.dataset[self.next_state_kdtree.query_radius(neighbor.centered_state.reshape(1, -1), neighbor.radius).item()])
        else:
            data_in_neighbourhood = deepcopy(self.dataset[self.distance(self.dataset.next_state, neighbor.centered_state) <= neighbor.radius])

        # Step 2: one-step backward expand
        if len(data_in_neighbourhood) == 0:
            return [], []
        else:
            return NeighbourSet(
                centered_state = data_in_neighbourhood.state,
                radius = np.minimum(
                    self.lipschitz_confidence, 
                    (neighbor.radius - self.distance(data_in_neighbourhood.next_state, neighbor.centered_state)) / self.lipschitz_fx(data_in_neighbourhood)
                ),
                visited=np.zeros(len(data_in_neighbourhood), dtype=bool),
            ), NeighbourSet(
                centered_state = data_in_neighbourhood.next_state,
                radius = neighbor.radius - self.distance(data_in_neighbourhood.next_state, neighbor.centered_state),
                visited=np.zeros(len(data_in_neighbourhood), dtype=bool),
            )
    
    def get_epsilon_controllable_set(self, state: np.ndarray):
        '''
        :param state (np.ndarray): the state to be tested
        '''
        assert self.epsilon_controllable_set == None, "The epsilon controllable list is not empty!"
        expand_counter = 0
        fig, ax = None, None
        self.plot_utils.set_orgin_state(state)

        self.epsilon_controllable_set = NeighbourSet.batch([NeighbourSet(state, self.epsilon)])
        # until all the neighbor sets are visited
        while not all([neighbor.visited for neighbor in self.epsilon_controllable_set]):
            idx_set = 0
            while idx_set < len(self.epsilon_controllable_set):
                neighbor = self.epsilon_controllable_set[idx_set]
                # TODO: more detailed implementation  
                if len(self.epsilon_controllable_set) == self.num_sample:
                    return
                if not neighbor.visited:
                    expand_neighbor_set, last_neighbor_set = self.backward_expand(neighbor)
                    self.epsilon_controllable_set[idx_set] = neighbor # set visited = True
                    for idx_expland, expand_neighbor in enumerate(expand_neighbor_set):
                        relation, idx_inlist = self.check_expand_neighbor_relation(expand_neighbor)
                        if relation == None:
                            self.epsilon_controllable_set.append(expand_neighbor)
                        elif relation == "set_in_expand":
                            # If you use fliter, only the elements after the current point are deleted, 
                            # Pro: overwritten neighbors are removed from the collection, 
                            # Con: the iteration starts from the beginning after the iteration ends.
                            # idx_inlist = list(filter(lambda x: x > idx_set, idx_inlist))

                            del self.epsilon_controllable_set[idx_inlist]
                            self.epsilon_controllable_set.append(expand_neighbor)
                        else:
                            assert relation == "expand_in_set", "relation is not correct!"
                        if relation != "expand_in_set" and self.plot_flag:
                            fig, ax = self.plot_utils.plot_backward(
                                state = expand_neighbor.centered_state, 
                                r = expand_neighbor.radius, 
                                next_state = last_neighbor_set[idx_expland].centered_state, 
                                next_r = last_neighbor_set[idx_expland].radius,
                                fig=fig,
                                ax=ax
                            )
                    if self.plot_flag and expand_counter%self.expand_plot_interval == 0:
                        self.plot_utils.plot_epsilon_controllable_list(self.epsilon_controllable_set, expand_counter)
                    expand_counter += 1
                    print("index in set: {}, expand count: {}, new_neighbor_num: {}, total_controllable_num: {}"
                        .format(idx_set, expand_counter, len(expand_neighbor_set), len(self.epsilon_controllable_set))
                    )
                idx_set += 1
        if self.plot_flag:
            self.plot_utils.plot_epsilon_controllable_list(self.epsilon_controllable_set, -1)
            self.plot_utils.save_figs(fig, ax)


    def run(self, state: np.ndarray):
        time_start = time.time()
        self.sample()
        time_sample = time.time() - time_start
        print("time for sampling: {:.4f}s".format(time_sample))

        time_plot = 0
        if self.plot_flag:
            self.plot_utils.plot_sample(self.buffer.buffer)
            time_plot = time.time() - time_start - time_sample
            print("time for plotting: {:.4f}s".format(time_plot))

        self.get_epsilon_controllable_set(state)
        time_calonestep = time.time() - time_start - time_sample - time_plot
        print("time for calculating epsilon controllable set: {:.4f}s".format(time_calonestep))

    def check_expand_neighbor_relation(self, expand_neighbor: NeighbourSet) -> Tuple[Optional[str], Optional[np.ndarray]]:
        dist = self.distance(expand_neighbor.centered_state, self.epsilon_controllable_set.centered_state)
        
        expand_in_set_condition = (dist <= self.epsilon_controllable_set.radius - expand_neighbor.radius)
        if np.any(expand_in_set_condition):
            return "expand_in_set", np.where(expand_in_set_condition)[0]
        
        list_in_expand_condition = (dist <= expand_neighbor.radius - self.epsilon_controllable_set.radius)
        if np.any(list_in_expand_condition):
            return "set_in_expand", np.where(list_in_expand_condition)[0]
        
        return None, None

    def lipschitz_fx(self, data: Transition) -> Union[float, np.ndarray]:
        # TODO: support batched data
        return np.ones(len(data))

        if self.use_kd_tree:
            data_in_neighbourhood = deepcopy(self.dataset[self.state_kdtree.query_radius(data.state.reshape(1, -1), self.lipschitz_confidence).item()])
        else:
            data_in_neighbourhood = deepcopy(self.dataset[self.distance(self.dataset.state, data.state) <= self.lipschitz_confidence])
        # only Lx
        # return max([(next_state - data.next_state) / (state - data.state) for data in data_in_neighbourhood])
        

        # Lx and Lu
        next_state_dist =[float(-distance) for distance in self.distance(data_in_neighbourhood.next_state, data.next_state)]
        states_dist = [[float(-self.distance(data_unit.state,data.state)),
                        float(-self.distance(data_unit.action,data.action))] for data_unit in data_in_neighbourhood]

        # solve QP: min Lx**2 + Lu**2, s.t. next_state_dist<=Lx*state_dist+Lu*action_dist
        time_start = time.time()
        P = matrix([[1.0, 0.0], [0.0, 1.0]])
        q = matrix([0.0, 0.0])
        h = matrix([next_state_dist])
        G = matrix(states_dist).T
        solution = solvers.qp(P, q, G, h)
        x = np.array(solution['x'])
        print(f'cost:qp_solving:{time.time() - time_start:.4f}s')
        return x[0]

    def lipschitz_fx_sampling(self, state: np.ndarray) -> np.ndarray:
        # calculate the lipschitz constant of the dynamics function at state within self.lipschitz_confidence
        unbatched = len(state.shape) == 1
        if unbatched == 1:
            states = states[None, :]
            actions = actions[None, :]
        batch_size, state_dim = states.shape
        Lx = np.zeros(batch_size)
        # sample state and action
        for _ in range(10):
            sample_action = self.env.action_space.sample()
            sample_state = np.random.randn(state_dim)
            sample_state = sample_state / np.linalg.norm(sample_state, ord=2) * np.random.uniform(0.0001, self.lipschitz_confidence)
            Lx = np.maximum(Lx, self.jacobi_atx(state + sample_state, sample_action))
        if unbatched:
            Lx = Lx[0]
        return Lx
    
    def jacobi_atx(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # calculate the lipschitz constant of the dynamics function at (state, action)
        # TODO: check whether this function support unbatched data
        unbatched = len(state.shape) == 1
        if unbatched:
            states = states[None, :]
            actions = actions[None, :]
        batch_size, state_dim = states.shape
        delta_d = 0.001
        lipschitz_x = np.zeros((batch_size, state_dim, state_dim))
        for i in range(state_dim):
            delta_x = np.eye(state_dim)[i] * delta_d
            lipschitz_x[:, :, i] = (
                self.env.model_forward(
                    states + delta_x, actions
                ) - self.env.model_forward(
                    states - delta_x, actions
                )
            ) / (2 * delta_d)
        result = np.linalg.norm(lipschitz_x, ord=2, axis=(1,2)) 
        if unbatched:
            result = result[0]
        return result
    
    @staticmethod
    def distance(state1: np.ndarray, state2: np.ndarray) -> float:
        if len(state1.shape) == 1 and len(state2.shape) == 1:
            return np.linalg.norm(state1 - state2, ord=2)
        else:
            return np.linalg.norm(state1 - state2, ord=2, axis=1)

    def clear(self):
        self.epsilon_controllable_set = None
        self.buffer.clear()
        self.dataset = None

    def seed(self, seed=None):
        self.env.seed(seed)
