# from re import X

# from Project.utils import plot_grid
from rrt_kino import *
import numpy as np
from sympy import *
from sympy.abc import x
from env_kino import NodeKino
import random #check if is used
from collections import deque
from utils import plot_kino, plot_grid
import time as timing
from scipy.stats import multivariate_normal

SEED = 666
random.seed(SEED)
np.random.seed(SEED)

# [50,65,0,0]

from scipy.stats import norm, multivariate_normal, bernoulli


class Informed_RRT_Star_Kino(RRT_Star_Kino):
    def __init__(
        self,
        env=None,
        x_start=[2, 2, 0, 0],
        x_goal=[98, 98, 0, 0],
        max_radius=100, #25, #init 100
        iter_max=50,
        state_dims=4,
        input_dims=2,
        state_limits=[[0, 100], [0, 100], [-10, 10], [-10, 10]],
        input_limits=[[-5, 5], [-5, 5]],
        stop_at=-np.inf,
    ):

        super().__init__(
            env,
            x_start,
            x_goal,
            max_radius,
            iter_max,
            state_dims,
            input_dims,
            state_limits,
            input_limits,
            stop_at,
        )

        self.name = "IRRTK_star"
        self.p_best = [self.x_start.node, self.x_goal.node]  # initial best path
        plt.ion()
        n_path = f'_N_{self.iter_max}'
        self.plotting_path = f'{self.name}{n_path}'
        self.sol = 0

    def init(self):
        self.t = Symbol("t")
        self.ts = Symbol("ts")
        self.x = Symbol("x")
        self.x0 = MatrixSymbol("x0", self.state_dims, 1)
        self.x1 = MatrixSymbol("x1", self.state_dims, 1)
        self.tau = Symbol("tau")

        # Distance
        G_ = lambdify(
            [self.tau, x],
            simplify(exp(self.A * (self.tau - x)))
            * self.B
            * self.R.inv()
            * self.B.T
            * simplify(exp(self.A.T * (self.tau - x))),
            "numpy",
        )
        x_bar_ = lambdify(
            [self.tau, self.ts],
            simplify(exp(self.A * (self.tau - self.ts))) * self.c,
            "numpy",
        )
        self.x_bar = lambdify(
            [self.tau, self.ts, self.x0],
            simplify(exp(self.A * self.tau)) * Matrix(self.x0)
            + integrate(Matrix(x_bar_(self.tau, self.ts)), (self.ts, 0, self.tau)),
            "numpy",
        )
        self.G = lambdify(
            [self.tau, x], integrate(Matrix(G_(self.tau, x)), (x, 0, self.tau)), "numpy"
        )
        self.d = lambdify(
            [self.x1, self.tau, self.ts, self.x0, x],
            Matrix(self.G(self.tau, x)).inv()
            * (Matrix(self.x1) - self.x_bar(self.tau, self.ts, self.x0)),
            "numpy",
        )

        # Tau*
        tau_star_ = (
            eye(1)
            - 2
            * (self.A * Matrix(self.x1) + self.c).T
            * self.d(self.x1, self.tau, self.ts, self.x0, x)
            - self.d(self.x1, self.tau, self.ts, self.x0, x).T
            * self.B
            * self.R
            * self.B.T
            * self.d(self.x1, self.tau, self.ts, self.x0, x)
        )
        self.tau_star = lambdify([self.x0, self.x1], tau_star_, "numpy")

        # Cost
        self.cost_tau = lambdify(
            [self.tau, x, self.x0, self.x1],
            Matrix([self.tau])
            + (self.x1 - self.x_bar(self.tau, self.ts, self.x0)).T
            * integrate(Matrix(G_(self.tau, x)), (x, 0, self.tau)).inv()
            * (self.x1 - self.x_bar(self.tau, self.ts, self.x0)),
            "numpy",
        )

        # States
        mat = Matrix(
            [
                [self.A, self.B * self.R.inv() * self.B.T],
                [zeros(self.state_dims, self.state_dims), -self.A.T],
            ]
        )
        exp_mat1 = lambdify([self.t, x], simplify(exp(mat * (self.t - x))), "numpy")
        exp_mat2 = lambdify(
            [self.tau, self.t], simplify(exp(mat * (self.t - self.tau))), "numpy"
        )
        solution_ = lambdify(
            [self.t, x],
            exp_mat1(self.t, x) * Matrix([[self.c], [zeros(self.state_dims, 1)]]),
            "numpy",
        )
        state0 = lambdify(
            [self.tau, self.t, self.ts, self.x0, self.x1, x],
            exp_mat2(self.tau, self.t)
            * Matrix(
                BlockMatrix(
                    [
                        [self.x1],
                        [Matrix(self.d(self.x1, self.tau, self.ts, self.x0, x))],
                    ]
                )
            ),
            "numpy",
        )
        sol = lambdify(
            [self.tau, self.t, x, self.x0, self.x1],
            state0(self.tau, self.t, self.ts, self.x0, self.x1, x)
            + integrate(Matrix(solution_(self.t, x)), (x, self.tau, self.t)),
            "numpy",
        )

        self.states = lambdify(
            [self.tau, self.t, x, self.x0, self.x1],
            Matrix(sol(self.tau, self.t, x, self.x0, self.x1))[0 : self.state_dims],
            "numpy",
        )

        # Control
        self.control_ = lambdify(
            [self.tau, self.t, x, self.x0, self.x1],
            self.R.inv()
            * self.B.T
            * simplify(exp(self.A.T * (self.tau - self.t)))
            * integrate(Matrix(G_(self.tau, x)), (x, 0, self.tau)).inv()
            * (Matrix(self.x1) - self.x_bar(self.tau, self.ts, self.x0)),
            "numpy",
        )

        cost, time = self.eval_cost(self.x_start.node, self.x_goal.node, time=None)
        cost = cost[0]

        self.x_start.cost = 0
        self.x_start.time = 0
        self.x_start.cost2goal = cost
        self.x_start.time2goal = time

        self.x_goal.cost = np.inf
        self.x_goal.time = np.inf
        self.x_goal.cost2goal = 0
        self.x_goal.time2goal = 0

        self.V.append(self.x_start)

        # The tranistion model defines how to move from sigma_current to sigma_new
        self.transition_model = lambda x: [
            np.random.normal(x[i], 5, (1,))[0] for i in range(len(x)) #0.5
        ]
        self.X_inf = [
            self.x_start.node,
            self.x_goal.node
        ]

    def planning(self):
        print("STARTING PLANNING")
        self.init()
        self.c_best = np.inf
        self.t_best = np.inf
        self.x_best = self.x_start
    
        i = 0
        while i < self.iter_max and self.c_best > self.stop_at:
            print(f"Iteration {i} #######################################")
            print(len(self.V))
            i += 1

            x_rand = self.Sample()
            x_rand, min_node, min_cost, min_time = self.ChooseParent(x_rand)

            if x_rand is None:
                continue

            x_rand = self.Rewire(x_rand)

            x_rand = self.isBest(x_rand, i) #modified for the GIF

            self.V.append(x_rand)

        print("self.c_best", self.c_best)
        self.path = self.ExtractPath(self.x_best)
        print(self.path)
        plot_kino(self, i, c_best=self.c_best[0], tau_star=self.t_best)
        plt.pause(2.01)
        animate(self)

    def eval_arrival_time(self, x0_, x1_):
        tau_star = Matrix(self.tau_star(x0_, x1_))

        p = Poly(tau_star[0]).all_coeffs()
        time_vec = np.roots(p)

        time = max([re(t) for t in time_vec if im(t) == 0 and re(t) >= 0])
        return 1 / time

    def eval_cost(self, x0, x1, time=None):

        if time == None:
            time = self.eval_arrival_time(x0, x1)

        cost = self.cost_tau(time, x, x0, x1)

        return cost, time

    def eval_states_and_inputs(self, x0, x1, time=None):

        if time == None:
            time = self.eval_arrival_time(x0, x1)

        states = lambdify([x, self.t], self.states(time, self.t, x, x0, x1))
        inputs = lambdify([x, self.t], Matrix(self.control_(time, self.t, x, x0, x1)))
        return states, inputs

    def isStateFree(self, states, t_init, t_goal):
        step = 0.3

        r = np.arange(t_init / step, t_goal / step) * step

        for time_step in r:
            state = states(self.t, time_step)
      
            for i in range(len(self.state_limits)):
                if (
                    state[i] < self.state_limits[i][0]
                    or state[i] > self.state_limits[i][1]
                ):
                    return False

            if self.isCollision(state):
                return False

        return True

    def isCollision(self, state):
        if self.env.is_inside_obs(state):
            return True
        return False


    def isInputFree(self, inputs, t_init, t_goal):
        resolution = 20
        step = (t_goal - t_init) / resolution

        r = np.arange(t_init / step, t_goal / step) * step

        for time_step in r:
            inp = inputs(self.t, time_step)

            for i in range(len(self.input_limits)):
                if inp[i] < self.input_limits[i][0] or inp[i] > self.input_limits[i][1]:
                    return False

        return True

    def Sample(self):
        delta = self.delta

        if self.c_best == np.inf:
            node = NodeKino(
                (
                    np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                    np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
                    np.random.uniform(-10, 10),
                    np.random.uniform(-10, 10),
                )
            )

            while self.isCollision(node.node):
                node = NodeKino(
                    (
                        np.random.uniform(
                            self.x_range[0] + delta, self.x_range[1] - delta
                        ),
                        np.random.uniform(
                            self.y_range[0] + delta, self.y_range[1] - delta
                        ),
                        np.random.uniform(-10, 10),
                        np.random.uniform(-10, 10),
                    )
                )

        else:
            node = self.MCMC_Sampling()

        return node

    def ChooseParent(self, x_rand):
        min_node = None

        min_cost = np.inf
        min_time = np.inf
        for node in self.V:
            cost, time = self.eval_cost(node.node, x_rand.node)
            cost = cost[0]
            
            if (
                cost < self.max_radius
                and node.cost + cost < min_cost
                and node.cost + cost < self.x_goal.cost
            ):

                states, inputs = self.eval_states_and_inputs(
                    node.node, x_rand.node, time
                )
              
                if self.isStateFree(states, 0, time) and self.isInputFree(
                    inputs, 0, time
                ):
                    print("EUREKA")
                    min_cost = cost + node.cost
                    min_time = time + node.time
                    min_node = node

        if min_node is None:
            return None, None, None, None

        # self.V[self.V.index(min_node)].cost = min_cost
        # self.V[self.V.index(min_node)].time = min_time
        self.V[self.V.index(min_node)].children.append(x_rand)
        x_rand.cost = min_cost
        x_rand.time = min_time
        x_rand.parent = self.V[self.V.index(min_node)]

        return x_rand, min_node, min_cost, min_time

    def Rewire(self, x_rand):
        stack = deque()
        stack.append((self.x_start, 0, 0))  # node, cost, time
        # ts = timing.time()
        while not len(stack) == 0:
            # print("LEN STACK")
            # print(len(stack))
            # print("#"*200)
            node, cost_imp, time_imp = stack.pop()

            node.cost -= cost_imp
            node.time -= time_imp
            if node.near_goal:
                if node.cost + node.cost2goal < self.c_best:
                    self.c_best = node.cost + node.cost2goal
                    self.t_best = node.time + node.time2goal
                    self.x_best = node
                continue

            diff = cost_imp
            time_diff = time_imp

            # per ogni nodo abbiamo il costo start-nodo
            # costo start-x_i (c'è) + x_i-nodo (da calcolare)
            # x_i diventa padre di un nodo ottimizzando il costo del path
            # ts1 = timing.time()
            partial_cost, partial_time = self.eval_cost(
                x_rand.node, node.node, time=None
            )
            # te1 = timing.time()
            # print("eval cost while:", te1-ts1)
            partial_cost = partial_cost[0]
            #print('Partial cost:', partial_cost)
            if partial_cost < self.max_radius:  # and self.isCollision()

                new_cost = partial_cost + x_rand.cost

                if new_cost < node.cost:
                    states, inputs = self.eval_states_and_inputs(
                        x_rand.node, node.node, partial_time
                    )
                    if self.isStateFree(states, 0, partial_time) and self.isInputFree(
                        inputs, 0, partial_time
                    ):

                        new_time = partial_time + x_rand.time
                        diff = node.cost - new_cost
                        time_diff = node.time - new_time

                        self.V[self.V.index(node)].parent = x_rand
                        self.V[self.V.index(node)].cost = new_cost
                        x_rand.children.append(node)

                        self.V[self.V.index(node)].time = new_time

            for child in node.children:
                stack.append((child, diff, time_diff))
        # te = timing.time()
        # print("stack time", te - ts)
        return x_rand

    def isBest(self, x_rand, i):
        cost, time = self.eval_cost(x_rand.node, self.x_goal.node)
        cost = cost[0]
        if x_rand.cost + cost < self.c_best:
            states, inputs = self.eval_states_and_inputs(
                x_rand.node, self.x_goal.node, time
            )
            if self.isStateFree(states, 0, time) and self.isInputFree(inputs, 0, time):
                self.c_best = x_rand.cost + cost
                self.t_best = x_rand.time + time
                self.x_goal.cost = self.c_best
                self.x_goal.time = self.t_best
                x_rand.near_goal = True
                x_rand.cost2goal = cost
                # x_rand.time2goal = time
                self.x_best = x_rand
                print("#" * 200)
                print("TROVATA SOLUZIONE")
                self.sol +=1
                print("#" * 200)
                self.p_best = self.ExtractPath(self.x_best)
                #print(self.X_inf)
                self.X_inf.extend([element for element in self.p_best if element not in self.X_inf])
                #print(self.X_inf)
                #self.X_inf = list(set(self.p_best) | set(self.X_inf))
                #self.X_inf = self.ExtractPath(self.x_best)
                self.path = self.ExtractPath(self.x_best)
                print(self.path)
                #print('c_best format:', self.c_best)
                plot_kino(self, i, c_best=self.c_best[0], tau_star=self.t_best)

        return x_rand

    def ExtractPath(self, node):
        path = [self.x_goal.node]

        while node.parent:
            # print(node.parent.x,node.parent.y)
            path.append(node.node)
            node = node.parent

        path.append(self.x_start.node)

        return path

    def in_informed(self, x):
        # check collision
        if self.isCollision(x.node):
            #print('Collide')
            return False

        # compute cost: x_start -> x -> x_goal
        cost1, _ = self.eval_cost(self.x_start.node, x.node)
        
        cost2, _ = self.eval_cost(x.node, self.x_goal.node)

        cost = cost1 + cost2

        if cost >= self.c_best:
            #print('non migliora')
            return False

        return True

    def MCMC_Sampling(self):  # x_i, c_best
        cont = True
        #print(self.X_inf)
        while 1:
            x_0 = self.X_inf[np.random.randint(0, len(self.X_inf))]
            while 1:
                x_next = self.Metropolis_Hastings(x_0, self.X_inf)
                x_next = NodeKino(x_next)

                if np.array_equal(np.array(x_next.node), np.array(x_0)):
                    continue
                else:
                    break

            if self.in_informed(x_next):
                break
            else:
                continue

        # print("x_next", x_next.node)
        #x_rand = NodeKino(x_next)
        return x_next

    def prior(self, x):
        # returns 1 for all valid values of the sample. Log(1) =0, so it does not affect the summation.
        # returns 0 for all invalid values of the sample (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
        # It makes the new sample infinitely unlikely.

        if self.state_limits[0][0] < x[0] < self.state_limits[0][1] and \
            self.state_limits[1][0] < x[1] < self.state_limits[1][1] and \
             self.state_limits[2][0] < x[2] < self.state_limits[2][1] and \
              self.state_limits[3][0] < x[3] < self.state_limits[3][1]:
            return 1
        return 0


    # Computes the likelihood of the data given a sigma (new or current) according to equation (2)
    def manual_log_like_normal(self, x, data):
        # data = the observation
        # return np.sum(-np.log(x[1] * np.sqrt(2* np.pi) )-((data-x[0])**2) / (2*x[1]**2))
        mean_data = np.mean(data, axis=0) # mean on the columns
        cov = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] #base value
        return np.log(multivariate_normal.pdf(
            np.array(x).flatten(), mean=mean_data.flatten(), cov=cov
        ))


    # Defines whether to accept or reject the new sample
    def acceptance(self, x, x_new):
        if x_new > x:
            return True
        else:
            accept = np.random.uniform(0, 1)
            # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
            # less likely x_new are less likely to be accepted
            return accept < (np.exp(x_new - x))

    def Metropolis_Hastings(self, start, data):
        x = start

        x_new = self.transition_model(x)
        x_lik = self.manual_log_like_normal(x, data)
        x_new_lik = self.manual_log_like_normal(x_new, data)
        if self.acceptance(
            x_lik + np.log(self.prior(x)), x_new_lik + np.log(self.prior(x_new))
        ):
            return x_new

        return x

    