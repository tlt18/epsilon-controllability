from copy import deepcopy
from typing import List
import os
import cvxopt

import matplotlib.pyplot as plt
import numpy as np

from buffer import Buffer
from env.massspring import MassSpring, MassSpringwoControl
from env.simpleocp import SimpleOCP, SimpleOCPwoControl
from env.pendulum import Pendulum, PendulumwoControl
from controllabilitytest import ControllabilityTest, Transition
from utils.timeit import Timeit

FILEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LipsTest(ControllabilityTest):
    def cal_linear_constraint(self, data: Transition) -> np.ndarray:
        # data is unbatched
        assert len(data.state.shape) == 1
        # find neighbours
        lipschitz_confidence = self.lipschitz_confidence
        while True:
            if self.use_kd_tree:
                data_in_neighbourhood = deepcopy(self.dataset[self.state_kdtree.query_radius(data.state.reshape(1, -1), lipschitz_confidence).item()])
            else:
                data_in_neighbourhood = deepcopy(self.dataset[self.distance(self.dataset.state, data.state) <= lipschitz_confidence])
            if len(data_in_neighbourhood) > 0:
                break
            else:
                lipschitz_confidence *= 2
        # compute coefficients
        next_state_dist = self.distance(data_in_neighbourhood.next_state, data.next_state)
        states_dist = self.distance(data_in_neighbourhood.state, data.state)
        actions_dist = self.distance(data_in_neighbourhood.action, data.action)
        print("lipschitz_confidence: ", lipschitz_confidence)
        print("sample_num: ", len(data_in_neighbourhood))
        # next_state_dist <= Lx * state_dist + Lu*action_dist
        return next_state_dist, states_dist, actions_dist
        
    def lipschitz_fx_optimizing_qp(self, data: Transition) -> np.ndarray:
        # estimate the lipschitz constant by QP
        unbatched = len(data.state.shape) == 1
        if unbatched:
            data = Transition.batch([data])

        lipschitz_x = np.zeros(len(data))
        lipschitz_u = np.zeros(len(data))
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
            lipschitz_u[idx] = solution['x'][1]

        if unbatched:
            lipschitz_x = lipschitz_x[0]
            lipschitz_u = lipschitz_u[0]
        return lipschitz_x, lipschitz_u
    
    def lipschitz_fx_sampling(self, data: Transition) -> np.ndarray:
        # calculate the lipschitz constant by sampling
        sample_num = 1000
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
        Lu = self.jacobi_atu(sample_states, actions).reshape(sample_num, batch_size)
        Lx = np.max(Lx, axis=0)
        Lu = np.max(Lu, axis=0)

        if unbatched:
            Lx = Lx[0]
            Lu = Lu[0]
        return Lx, Lu
            
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
    
    def jacobi_atu(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        # calculate the lipschitz constant of the dynamics function at (state, action)
        unbatched = len(state.shape) == 1
        if unbatched:
            states = state[None, :]
            actions = action[None, :]
        else:
            states = state
            actions = action
        batch_size, action_dim = actions.shape
        delta_d = 0.001
        lipschitz_u = np.zeros((batch_size, action_dim, action_dim))
        delta_u = np.eye(action_dim) * delta_d
        delta_u = delta_u[None, :]
        states = states[:, None, :].repeat(action_dim, axis=1).reshape(-1, self.env.observation_space.shape[0])
        actions = actions[:, None, :]
        lipschitz_u = (
            self.env.model_forward(
                states, (actions + delta_u).reshape(-1, action_dim)
            ) - self.env.model_forward(
                states, (actions - delta_u).reshape(-1, action_dim)
            )
        ) / (2 * delta_d)
        lipschitz_u = lipschitz_u.reshape(batch_size, action_dim, self.env.observation_space.shape[0])
        result = np.linalg.norm(lipschitz_u, ord=2, axis=(1,2)) 
        if unbatched:
            result = result[0]
        return result
        
    
    @staticmethod
    def distance(state1: np.ndarray, state2: np.ndarray) -> float:
        if len(state1.shape) == 1 and len(state2.shape) == 1:
            return np.linalg.norm(state1 - state2, ord=2)
        else:
            return np.linalg.norm(state1 - state2, ord=2, axis=1)



if __name__ == "__main__":
    num_sample = 1000
    epsilon = 0.05
    target_state = np.array([0.0, 0.0])
    lipschitz_confidence = 0.2

    env_name = "MassSpring"
    env = eval(env_name)(seed = 1)
    buffer = Buffer(buffer_size = num_sample)
    test = LipsTest(
        env = env,
        buffer = buffer,
        target_state = target_state,
        epsilon = epsilon, 
        num_sample = num_sample,
        lipschitz_confidence = lipschitz_confidence,
        use_kd_tree = True,
        lips_estimate_mode = "sampling",
        expand_plot_interval = 1000, 
        backward_plot_interval = 10000000000,
        plot_expand_flag = False,
        plot_backward_flag = False,
    )
    test.sample()

    # Lipschitz constants @(state, action, next_state)
    state = np.array([-0.1, 0.05])
    action = env.action_space.sample()
    next_state = env.model_forward(state, action)
    transition = Transition(state, action, next_state)

    lips_by_opt_qp_x, lips_by_opt_qp_u = test.lipschitz_fx_optimizing_qp(transition)
    next_state_dist, states_dist, actions_dist = test.cal_linear_constraint(transition)
    lips_by_sampling_x, lips_by_sampling_u = test.lipschitz_fx_sampling(transition)

    # plot next_state_dist = X * state_dist + Y * action_dist in (X,Y) plane
    for i in range(len(next_state_dist)):
        if actions_dist[i] < 0.00001:
            plt.axvline(next_state_dist[i] / states_dist[i])
        X = np.linspace(0, next_state_dist[i] / states_dist[i], 100)
        Y = (next_state_dist[i] - X * states_dist[i]) / actions_dist[i]
        plt.plot(X, Y, color='#00B0F0')
    plt.plot(X, Y, color='#00B0F0', label = "linear constraint")

    # plot 1/4 circle: X^2 + Y^2 = lips_by_opt_qp_x ** 2 + lips_by_opt_qp_u ** 2
    X = np.linspace(0, np.sqrt(lips_by_opt_qp_x ** 2 + lips_by_opt_qp_u ** 2), 100)
    Y = np.sqrt(lips_by_opt_qp_x ** 2 + lips_by_opt_qp_u ** 2 - X ** 2)
    plt.plot(X, Y, color='#F59D56', label = "objective function", linewidth=2)

    # plot possible Lips cone
    max_lips = max(lips_by_opt_qp_x, lips_by_opt_qp_u)
    plt.plot([lips_by_sampling_x, lips_by_sampling_x], [lips_by_sampling_u, 2 * max_lips], color='#00B050', label = "Lips cone")
    plt.plot([lips_by_sampling_x, 2 * max_lips], [lips_by_sampling_u, lips_by_sampling_u], color='#00B050')
    
    # plot point (lips_by_opt_qp_x, lips_by_opt_qp_u)
    plt.text(lips_by_opt_qp_x + 0.1, lips_by_opt_qp_u + 0.1, f"({lips_by_opt_qp_x:.2f}, {lips_by_opt_qp_u:.2f})", color='#F59D56', fontsize=12)
    plt.scatter(lips_by_opt_qp_x, lips_by_opt_qp_u, marker='*', color='#F59D56', s=100, zorder=3)
    plt.text(lips_by_sampling_x + 0.3, lips_by_sampling_u + 0.3, f"({lips_by_sampling_x:.2f}, {lips_by_sampling_u:.2f})", color='#00B050', fontsize=12)

    plt.legend(loc='upper right')
    plt.xlim([-0.1, 2 * max_lips + 0.1])
    plt.ylim([-0.1, 2 * max_lips + 0.1])
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title(f"num_sample: {num_sample}")
    plt.xlabel("Lx")
    plt.ylabel("Lu")

    plt.savefig(os.path.join(FILEPATH, "figs", "lipschitz", env_name + str(num_sample) + "lipschitz.png"), bbox_inches='tight', pad_inches=0.2)

