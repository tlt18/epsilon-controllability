# controllability test for dynamics systems in datative setting
import random
from copy import deepcopy
from dataclasses import dataclass
import datetime
import os
from typing import List, Optional, Tuple, Union

import cvxopt
import gym
import numpy as np
from sklearn.neighbors import KDTree

from buffer import Buffer
from utils.utils_plots import PlotUtils, FILEPATH
from utils.timeit import timeit, Timeit


cvxopt.solvers.options['show_progress'] = False


@dataclass
class NeighbourSet:
    centered_state: np.ndarray
    radius: Union[float, np.ndarray]
    visited: Union[bool, np.ndarray] = False
    controllable_steps: Union[int, np.ndarray] = 0

    @staticmethod
    def batch(neighbor_list: List["NeighbourSet"]) -> "NeighbourSet":
        return NeighbourSet(
            centered_state = np.array([neighbor.centered_state for neighbor in neighbor_list]),
            radius = np.array([neighbor.radius for neighbor in neighbor_list]),
            visited = np.array([neighbor.visited for neighbor in neighbor_list]),
            controllable_steps = np.array([neighbor.controllable_steps for neighbor in neighbor_list]),
        )
    
    def __len__(self):
        assert len(self.radius.shape) > 0, "The data must be batched!"
        assert len(self.centered_state) == len(self.radius) == len(self.visited) == len(self.controllable_steps), \
            "The length of centered_state, radius and visited are not equal!"
        return len(self.centered_state)
    
    def replace(self, idx_1: int, idx_2: int):
        self.centered_state[[idx_1, idx_2]] = self.centered_state[[idx_2, idx_1]]
        self.radius[[idx_1, idx_2]] = self.radius[[idx_2, idx_1]]
        self.visited[[idx_1, idx_2]] = self.visited[[idx_2, idx_1]]
        self.controllable_steps[[idx_1, idx_2]] = self.controllable_steps[[idx_2, idx_1]]
    
    def __getitem__(self, index):
        return NeighbourSet(
            centered_state = self.centered_state[index],
            radius = self.radius[index],
            visited = self.visited[index],
            controllable_steps = self.controllable_steps[index],
        )
    
    def __setitem__(self, index, neighbor: "NeighbourSet"):
        self.centered_state[index] = neighbor.centered_state
        self.radius[index] = neighbor.radius
        self.visited[index] = neighbor.visited
        self.controllable_steps[index] = neighbor.controllable_steps
    
    def __delitem__(self, index):
        self.centered_state = np.delete(self.centered_state, index, axis=0)
        self.radius = np.delete(self.radius, index, axis=0)
        self.visited = np.delete(self.visited, index, axis=0)
        self.controllable_steps = np.delete(self.controllable_steps, index, axis=0)

    def append(self, neighbor: "NeighbourSet"):
        self.centered_state = np.append(self.centered_state, neighbor.centered_state.reshape(1, -1), axis=0)
        self.radius = np.append(self.radius, neighbor.radius.reshape(1), axis=0)
        self.visited = np.append(self.visited, neighbor.visited.reshape(1), axis=0)
        self.controllable_steps = np.append(self.controllable_steps, neighbor.controllable_steps.reshape(1), axis=0)

@dataclass    
class Transition:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    lipschitz_x: Optional[np.ndarray] = None
    is_controllable: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.is_controllable is None:
            self.is_controllable = np.full(self.state.shape[0], False)

    def __len__(self):
        assert len(self.state.shape) > 0, "The data must be batched!"
        assert len(self.state) == len(self.action) == len(self.next_state), \
            "The length of state, action and next_state are not equal!"
        return len(self.state)
    
    def __getitem__(self, index):
        return Transition(
            state = self.state[index],
            action = self.action[index],
            next_state = self.next_state[index],
            lipschitz_x = self.lipschitz_x[index] if self.lipschitz_x is not None else None,
            is_controllable = self.is_controllable[index],
        )

    @staticmethod
    def batch(transitionList: List["Transition"]) -> "Transition":
        return Transition(
            state = np.array([transition.state for transition in transitionList]),
            action = np.array([transition.action for transition in transitionList]),
            next_state = np.array([transition.next_state for transition in transitionList]),
            lipschitz_x = np.array([transition.lipschitz_x for transition in transitionList]) if transitionList[0].lipschitz_x is not None else None,
            is_controllable = np.array([transition.is_controllable for transition in transitionList]),
        )


class  ControllabilityTestforAll:
    def __init__(
            self, 
            env: gym.Env , 
            buffer: Buffer, 
            target_state: np.ndarray,
            epsilon: float = 0.1,
            num_sample: int = 10000,
            lipschitz_confidence: float = 0.2,
            use_kd_tree: bool = False,
            lips_estimate_mode: str = "sampling",
            expand_plot_interval: int = 1, 
            backward_plot_interval: int = 100,
            plot_expand_flag: bool = True,
            plot_backward_flag: bool = False,

        ):
        self.env = env
        self.buffer = buffer
        self.num_sample = num_sample
        self.target_state = target_state.astype(np.float32)
        self.epsilon = epsilon
        self.lipschitz_confidence = lipschitz_confidence
        self.use_kd_tree = use_kd_tree
        self.expand_plot_interval = expand_plot_interval
        self.plot_expand_flag = plot_expand_flag
        self.plot_backward_flag = plot_backward_flag
        self.epsilon_controllable_set: NeighbourSet = None
        self.lipschitz_fx = getattr(self, f"lipschitz_fx_{lips_estimate_mode}")

        self.fig_title = f"{env.__class__.__name__}/{target_state}state-{epsilon}epsilon-{num_sample}samples-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.plot_utils: PlotUtils = PlotUtils(
            obs_space = self.env.observation_space, 
            action_space = self.env.action_space,
            orgin_state = self.target_state,
            orgin_radius = self.epsilon,
            fig_title = self.fig_title,
            backward_plot_interval = backward_plot_interval,
        )
        self.dataset = None
        self.state_kdtree: KDTree = None
        self.next_state_kdtree: KDTree = None
        self.sample_flag = False
    def sample(self):
        random.seed(2024)
        state = self.env.reset()
        for _ in range(self.num_sample):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.buffer.add((state, action, reward, next_state, done))
            # print(state, action, reward, next_state)
            state = self.env.reset() if done else next_state
        self.dataset = Transition(*self.buffer.get_data())
        if self.use_kd_tree:
            self.state_kdtree = KDTree(self.dataset.state, leaf_size=40, metric='euclidean')
            self.next_state_kdtree = KDTree(self.dataset.next_state, leaf_size=40, metric='euclidean')

    def backward_expand(self, neighbor: NeighbourSet, center_only: bool = False) -> Tuple[NeighbourSet, NeighbourSet]:
        '''
        :param neighbor (NeighbourSet): the neighbor set to be expanded
        :param center_only (bool): get the state in dataset that is in the neighborhood of the center state only
        :return (Tuple[List[NeighbourSet], List[NeighbourSet]]): the expanded neighbor set and last neighbor set
        '''
        assert not neighbor.visited, "The neighbor set has been visited!"
        neighbor.visited = True
        
        # Step 1: Find all the states in the buffer that belong to the neighborhood set
        if center_only:
            radius_query = 0.0
        else:
            radius_query = neighbor.radius

        if self.use_kd_tree:
            idx_neighbourhood = self.next_state_kdtree.query_radius(neighbor.centered_state.reshape(1, -1), radius_query).item()
        else:
            idx_neighbourhood = self.distance(self.dataset.next_state, neighbor.centered_state) <= radius_query
        data_in_neighbourhood = deepcopy(self.dataset[idx_neighbourhood])
        self.dataset.is_controllable[idx_neighbourhood] = np.ones_like(self.dataset.is_controllable[idx_neighbourhood], dtype=bool)

        # Step 2: one-step backward expand
        if len(data_in_neighbourhood) == 0:
            return [], []
        else:
            r_state = np.minimum(
                    self.lipschitz_confidence, 
                    (neighbor.radius - self.distance(data_in_neighbourhood.next_state, neighbor.centered_state)) /\
                    data_in_neighbourhood.lipschitz_x
            )
            r_next_state = neighbor.radius - self.distance(data_in_neighbourhood.next_state, neighbor.centered_state)
            # For loose mode:
            # r_state = self.epsilon * np.ones(len(data_in_neighbourhood))
            # r_next_state = self.epsilon * np.ones(len(data_in_neighbourhood))
            return NeighbourSet(
                centered_state = data_in_neighbourhood.state,
                radius = r_state,
                visited = np.zeros(len(data_in_neighbourhood), dtype=bool),
                controllable_steps = (neighbor.controllable_steps + 1) * np.ones(len(data_in_neighbourhood)),
            ), NeighbourSet(
                centered_state = data_in_neighbourhood.next_state,
                radius = r_next_state,
                visited=np.zeros(len(data_in_neighbourhood), dtype=bool),
                controllable_steps = -1 * np.ones(len(data_in_neighbourhood)),
            )

    def get_epsilon_controllable_set(self, state: np.ndarray, epsilon: float):
        '''
        :param state (np.ndarray): the state to be tested
        '''
        assert self.epsilon_controllable_set == None, "The epsilon controllable list is not empty!"
        expand_counter = 0
        fig, ax = None, None

        self.epsilon_controllable_set = NeighbourSet.batch([NeighbourSet(state, epsilon)])
        # until all the neighbor sets are visited
        while not all([neighbor.visited for neighbor in self.epsilon_controllable_set]):
            idx_set = 0
            new_neighbor_num = 0
            while idx_set < len(self.epsilon_controllable_set):
                # find the neighbor with the largest radius
                idx_max = np.argmax(self.epsilon_controllable_set.radius[idx_set:])
                self.epsilon_controllable_set.replace(idx_set, idx_set + idx_max)

                neighbor = self.epsilon_controllable_set[idx_set]
                if not neighbor.visited:
                    # expand_neighbor_set, last_neighbor_set = self.backward_expand(neighbor, center_only=(idx_set!=0))
                    expand_neighbor_set, last_neighbor_set = self.backward_expand(neighbor)
                    self.epsilon_controllable_set[idx_set] = neighbor # set visited = True
                    for idx_expland, expand_neighbor in enumerate(expand_neighbor_set):
                        relation, idx_inlist = self.check_expand_neighbor_relation(expand_neighbor)
                        if relation == None:
                            self.epsilon_controllable_set.append(expand_neighbor)
                            new_neighbor_num += 1
                        elif relation == "set_in_expand":
                            # If you use fliter, only the elements after the current point are deleted.
                            # Pro: only one iteration.
                            # Con: repeated points stay in the set.
                            idx_dellist = list(filter(lambda x: x > idx_set, idx_inlist))
                            del self.epsilon_controllable_set[idx_dellist]
                            self.epsilon_controllable_set.append(expand_neighbor)
                            new_neighbor_num += 1
                        else:
                            assert relation == "expand_in_set" or "same_state", "relation is not correct!"
                        if relation != "expand_in_set" and relation != "same_state" and self.plot_backward_flag:
                            fig, ax = self.plot_utils.plot_backward(
                                state = expand_neighbor.centered_state, 
                                r = expand_neighbor.radius, 
                                next_state = last_neighbor_set[idx_expland].centered_state, 
                                next_r = last_neighbor_set[idx_expland].radius,
                                fig=fig,
                                ax=ax
                            )
                    if self.plot_expand_flag and expand_counter%self.expand_plot_interval == 0:
                        pass
                        # self.plot_utils.plot_epsilon_controllable_set(self.epsilon_controllable_set, expand_counter, self.dataset, self.target_state)
                    expand_counter += 1
                    if idx_set % 100 == 0 or idx_set == len(self.epsilon_controllable_set) - 1:
                        # print("index in set: {}, new_neighbor_num: {}, total_controllable_num: {}, current radius: {}"
                        #     .format(idx_set, new_neighbor_num, len(self.epsilon_controllable_set), neighbor.radius)
                        # )
                        new_neighbor_num = 0
                idx_set += 1
        if self.plot_backward_flag and fig is not None and ax is not None:
            self.plot_utils.save_figs(fig, ax)
        if self.plot_expand_flag:
            pass
            # self.plot_utils.plot_epsilon_controllable_set(self.epsilon_controllable_set, expand_counter, self.dataset, self.target_state)

    def estimate_lipschitz_constant(self):
        self.dataset.lipschitz_x = self.lipschitz_fx(self.dataset)

    def run(self):
        # if self.plot_expand_flag:
        #
        #     os.makedirs(FILEPATH + f"/figs/{self.fig_title}/epsilon_controllable_set", exist_ok=True)
        if self.plot_backward_flag:
            os.makedirs(FILEPATH + f"/figs/{self.fig_title}/expand_backward", exist_ok=True)

        with Timeit("sample time"):
            self.sample()

        with Timeit("estimate lipschitz constant time"):
            self.estimate_lipschitz_constant()

        with Timeit("calculate epsilon controllable set time"):
            self.get_epsilon_controllable_set(self.target_state, self.epsilon)

        with Timeit("count controllable states"):
            controllable_num, proportion = self.count_states()
            file = open(FILEPATH + f"/figs/{self.env.__class__.__name__}/count.txt", "a")
            # file.write("controllable_num: "+str(controllable_num)+" proportion: "+str(proportion)+"\n")
            file.write(f"{self.epsilon} " + str(proportion)+"\n")
            file.close()
        # with Timeit("plot.py sample time"):
        #     # os.makedirs(FILEPATH + f"/figs/{self.fig_title}", exist_ok=True)
        #     # self.plot_utils.plot_sample(self.dataset)
        #     # self.plot_utils.plot_controllable_data(self.dataset)
        
        # self.save_epsilon_controllable_set()

    def count_states(self):
        controllable_num = np.sum(self.dataset.is_controllable==True)
        return controllable_num, controllable_num/len(self.dataset)
    def save_epsilon_controllable_set(self):
        epsilon_controllable_set = np.concatenate(
            [
                self.epsilon_controllable_set.centered_state,
                self.epsilon_controllable_set.radius.reshape(-1, 1),
                self.epsilon_controllable_set.visited.reshape(-1, 1),
                self.epsilon_controllable_set.controllable_steps.reshape(-1, 1),
            ], axis=1
        )
        header = ', '.join(["state"+str(i) for i in range(self.env.observation_space.shape[0])] + ["radius", "visited", "controllable steps"])
        np.savetxt(FILEPATH + f"/figs/{self.fig_title}/epsilon_controllable_set.csv", epsilon_controllable_set, delimiter=",", header=header, comments="")

    def check_expand_neighbor_relation(self, expand_neighbor: NeighbourSet) -> Tuple[Optional[str], Optional[np.ndarray]]:
        dist = self.distance(expand_neighbor.centered_state, self.epsilon_controllable_set.centered_state)
        
        expand_in_set_condition = (dist <= self.epsilon_controllable_set.radius - expand_neighbor.radius)
        if np.any(expand_in_set_condition):
            return "expand_in_set", np.where(expand_in_set_condition)[0]
        
        # same_state_condition = dist < 1e-6
        # if np.any(same_state_condition):
        #     return "same_state", np.where(same_state_condition)[0]
        
        list_in_expand_condition = (dist <= expand_neighbor.radius - self.epsilon_controllable_set.radius)
        if np.any(list_in_expand_condition):
            return "set_in_expand", np.where(list_in_expand_condition)[0]
        
        return None, None

    def lipschitz_fx_optimizing_qp(self, data: Transition) -> np.ndarray:
        # estimate the lipschitz constant by QP
        unbatched = len(data.state.shape) == 1
        if unbatched:
            data = Transition.batch([data])

        lipschitz_x = np.zeros(len(data))
        for idx, single_transition in enumerate(data):
            lipschitz_confidence = self.lipschitz_confidence
            while True:
                if self.use_kd_tree:
                    data_in_neighbourhood = deepcopy(self.dataset[self.state_kdtree.query_radius(single_transition.state.reshape(1, -1), lipschitz_confidence).item()])
                else:
                    data_in_neighbourhood = deepcopy(self.dataset[self.distance(self.dataset.state, single_transition.state) <= lipschitz_confidence])
                if len(data_in_neighbourhood) > 0:
                    break
                else:
                    lipschitz_confidence *= 2

            next_state_negdist = (- self.distance(data_in_neighbourhood.next_state, data.next_state))
            states_negdist = - self.distance(data_in_neighbourhood.state, data.state)
            actions_negdist = - self.distance(data_in_neighbourhood.action, data.action)
            concat_negdist = np.stack([states_negdist, actions_negdist], axis = 0)

            # solve QP: min Lx**2 + Lu**2, s.t. next_state_dist <= Lx * state_dist + Lu*action_dist
            P = cvxopt.matrix([
                [1.0, 0.0], 
                [0.0, 1.0]
            ])
            q = cvxopt.matrix([0.0, 0.0])
            h = cvxopt.matrix(next_state_negdist.astype(np.double))
            G = cvxopt.matrix(concat_negdist.astype(np.double)).T
            '''
            minimize    (1/2)*x'*P*x + q'*x
            subject to  G*x <= h
                        A*x = b.
            '''
            solution = cvxopt.solvers.qp(P, q, G, h)
            lipschitz_x[idx] = solution['x'][0]

        if unbatched:
            lipschitz_x = lipschitz_x[0]
        return lipschitz_x
    
    def lipschitz_fx_optimizing_lp(self, data: Transition) -> np.ndarray:
        # estimate the lipschitz constant by LP
        unbatched = len(data.state.shape) == 1
        if unbatched:
            data = Transition.batch([data])

        lipschitz_x = np.zeros(len(data))
        for idx, single_transition in enumerate(data):
            lipschitz_confidence = self.lipschitz_confidence
            while True:
                if self.use_kd_tree:
                    data_in_neighbourhood = deepcopy(self.dataset[self.state_kdtree.query_radius(single_transition.state.reshape(1, -1), lipschitz_confidence).item()])
                else:
                    data_in_neighbourhood = deepcopy(self.dataset[self.distance(self.dataset.state, single_transition.state) <= lipschitz_confidence])
                if len(data_in_neighbourhood) > 0:
                    break
                else:
                    lipschitz_confidence *= 2

            next_state_negdist = (- self.distance(data_in_neighbourhood.next_state, data.next_state))
            states_negdist = - self.distance(data_in_neighbourhood.state, data.state)
            actions_negdist = - self.distance(data_in_neighbourhood.action, data.action)
            concat_negdist = np.stack([states_negdist, actions_negdist], axis = 0)

            # solve LP: min Lx + Lu, s.t. next_state_dist <= Lx * state_dist + Lu*action_dist
            C = cvxopt.matrix(
                [1.0, 1.0])
            h = cvxopt.matrix(next_state_negdist.astype(np.double))
            G = cvxopt.matrix(concat_negdist.astype(np.double)).T
            '''
            minimize     C*x
            subject to  G*x <= h
            
            '''
            solution = cvxopt.solvers.lp(C, G, h)
            lipschitz_x[idx] = solution['x'][0]

        if unbatched:
            lipschitz_x = lipschitz_x[0]
        return lipschitz_x

    def lipschitz_fx_maxdistance(self, data: Transition) -> np.ndarray:
        # estimate the lipschitz constant by max dist
        # L := max(d(f(x1,u1), f(x2,u2)) / max(d(x1,x2), d(u1,u2)))
        # then we have
        # d(f(x1,u1), f(x2,u2)) <= L * max(d(x1,x2), d(u1,u2)) <= L * d(x1, x2) + L * d(u1, u2)
        # on the other hand,
        # d(f(x1,u1), f(x2,u2)) <= min Lx * d(x1, x2) + min Lu * d(u1, u2)
        # min L = min Lx + min Lu
        unbatched = len(data.state.shape) == 1
        if unbatched:
            data = Transition.batch([data])

        lipschitz_x = np.zeros(len(data))
        for idx, single_transition in enumerate(data):
            lipschitz_confidence = self.lipschitz_confidence
            while True:
                if self.use_kd_tree:
                    data_in_neighbourhood = deepcopy(self.dataset[self.state_kdtree.query_radius(single_transition.state.reshape(1, -1), lipschitz_confidence).item()])
                else:
                    data_in_neighbourhood = deepcopy(self.dataset[self.distance(self.dataset.state, single_transition.state) <= lipschitz_confidence])
                if len(data_in_neighbourhood) > 0:
                    break
                else:
                    lipschitz_confidence *= 2
                    
            lipschitz_x[idx] = np.max(
                self.distance(single_transition.next_state, data_in_neighbourhood.next_state) / \
                np.max([
                    self.distance(single_transition.state, data_in_neighbourhood.state), 
                    self.distance(single_transition.action, data_in_neighbourhood.action)
                ], axis=0)
            )

        if unbatched:
            lipschitz_x = lipschitz_x[0]
        return lipschitz_x

    def lipschitz_fx_sampling(self, data: Transition) -> np.ndarray:
        # calculate the lipschitz constant by sampling
        sample_num = 10
        unbatched = len(data.state.shape) == 1
        if unbatched:
            data = Transition.batch([data])
        batch_size, state_dim = data.state.shape
        # sample state
        sample_delta_states = np.random.randn(sample_num, batch_size, state_dim)
        sample_delta_states = sample_delta_states / \
            np.linalg.norm(sample_delta_states, axis=2, keepdims=True) * \
            np.random.uniform(0.00001, self.lipschitz_confidence, size=(sample_num, batch_size, 1))
        # sample_delta_states: [sample_num, batch_size, state_dim]
        # states: [batch_size, state_dim]
        sample_states = (sample_delta_states + data.state[None, :]).reshape(-1, state_dim)
        actions = data.action[None, :].repeat(sample_num, axis=0).reshape(-1, self.env.action_space.shape[0])

        Lx = self.jacobi_atx(sample_states, actions).reshape(sample_num, batch_size)
        Lx = np.max(Lx, axis=0)

        if unbatched:
            Lx = Lx[0]
        return Lx
            
    def jacobi_atx(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # calculate the lipschitz constant of the dynamics function at (state, action)
        unbatched = len(state.shape) == 1
        if unbatched:
            states = state[None, :]
            actions = action[None, :]
        else:
            states = state
            actions = action
        batch_size, state_dim = states.shape
        delta_d = 0.001
        lipschitz_x = np.zeros((batch_size, state_dim, state_dim))
        delta_x = np.eye(state_dim) * delta_d
        delta_x = delta_x[None, :]
        states = states[:, None, :]
        actions = actions[:, None, :].repeat(state_dim, axis=1).reshape(-1, self.env.action_space.shape[0])
        lipschitz_x = (
            self.env.model_forward(
                (states + delta_x).reshape(-1, state_dim), actions
            ) - self.env.model_forward(
                (states - delta_x).reshape(-1, state_dim), actions
            )
        ) / (2 * delta_d)
        lipschitz_x = lipschitz_x.reshape(batch_size, state_dim, state_dim)
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
