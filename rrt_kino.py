
from rrt_star import *
import numpy as np
from sympy import *
from sympy.abc import x
from env_kino import NodeKino, EnvKino
import random
from collections import deque
from utils import animate, plot_kino, create_dir
import time as timing
import matlab.engine
import matplotlib
matplotlib.use('Agg')

class RRT_Star_Kino(RRT_Star):
    def __init__(self, env = None, x_start = [2,2,0,0], x_goal = [50,65,0,0], max_radius = 100,
                 iter_max = 50, state_dims = 4, input_dims = 2, state_limits = [[0, 100], [0, 100], [-10, 10], [-10, 10]],
                 input_limits = [[-5, 5], [-5, 5]], stop_at = -np.inf, custom_env = False, seed = 666, ppath="./simulations"):

        self.name = 'RRTK_star'
        self.state_dims = state_dims
        self.input_dims = input_dims
        self.stop_at = stop_at
        self.custom_env = custom_env
        self.seed = seed

        self.A = zeros(self.state_dims,self.state_dims)
        self.A[0,2] = 1.0
        self.A[1,3] = 1.0
        self.B = zeros(self.state_dims, self.input_dims)
        self.B[2,0] = 1.0
        self.B[3,1] = 1.0
        self.R = eye(self.input_dims)
        self.c = zeros(self.state_dims,1)
        self.x_start = NodeKino(x_start)
        self.x_goal = NodeKino(x_goal)

        self.state_limits = state_limits
        self.input_limits = input_limits


        self.iter_max = iter_max
        if self.stop_at > 0:
            self.iter_max = 8000
        self.V = []
        self.max_radius = max_radius
        self.env = env
        self.x_range = self.state_limits[0]
        self.y_range = self.state_limits[1]
        
        self.fig, self.ax = plt.subplots()
        self.delta = self.env.delta
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        plt.ion()
        custom_path = '_customEnv' if self.custom_env else ''
        c_path = f'_C_{self.stop_at}' if self.stop_at!=0 else ''
        n_path = f'_N_{self.iter_max}' if self.stop_at==0 else ''
        s_path = f'_S_{self.seed}'
        self.plotting_path = f'{self.name}{n_path}{c_path}{custom_path}{s_path}'
        self.sol = 0
        self.firstsol = None
        self.ppath = ppath


    def init(self):
        # create_dir(self.ppath)
        # with open(f"{self.ppath}/{self.name}_{self.seed}.tsv", "w") as f:
        #     f.write("It\tC_best\tT_best\tTime\tN_nodi\tB_path\tSol\n")

        self.eng = matlab.engine.start_matlab()
        self.matlab = self.eng.eqs(self.state_dims, self.input_dims, self.state_limits, self.input_limits, self.max_radius, np.array(self.env.obs_rectangle, dtype=np.float64))

        
        self.t = Symbol('t')
        self.ts = Symbol('ts')
        self.x = Symbol('x')
        self.x0 = MatrixSymbol('x0',self.state_dims,1)
        self.x1 = MatrixSymbol('x1',self.state_dims,1)
        self.tau = Symbol('tau')

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

        self.x_start.cost = 0
        self.x_start.time = 0
        self.x_start.cost2goal = cost
        self.x_start.time2goal = time

        self.x_goal.cost = np.inf
        self.x_goal.time = np.inf
        self.x_goal.cost2goal = 0
        self.x_goal.time2goal = 0

        self.V.append(self.x_start)

    def planning(self):
        print('STARTING PLANNING')
        self.init()
        self.c_best = np.inf
        self.t_best = np.inf
        self.x_best = self.x_start
   
        sol = 0
        i = 0
        while i<self.iter_max and self.c_best > self.stop_at:
            print(f"Iteration {i} #######################################")
            # print(len(self.V))
            ts = timing.time()
            i += 1

            x_rand = self.Sample()

            # ChooseParent: given x_rand choose the best parent, i.e. best cost
            x_rand, min_node, min_cost, min_time  = self.ChooseParent(x_rand)

            if x_rand is None:
                continue

            # Insert x_rand in the path tree        
            self.V.append(x_rand)

            # Rewire: x_rand becomes parent of the nodes such that
            # the cost(x_start->x_rand->node) < cost(x_start->node)
            x_rand = self.Rewire(x_rand)       

            # with open(f"{self.ppath}/{self.name}_{self.seed}.tsv", "a") as f:
            #     if self.sol > sol:
            #         self.path = self.ExtractPath(self.x_best)
            #         plot_kino(self, i, c_best=self.c_best, tau_star=self.t_best)
            sol = self.sol

            # isBest: check whether the path cost from x_start->x_goal 
            # passing through x_rand is better than the previous best path cost 
            x_rand =  self.isBest(x_rand, i)
            te = timing.time()
            # with open(f"{self.ppath}/{self.name}_{self.seed}.tsv", "a") as f:
            #     if self.sol > sol:
            #             b_path = [elem.tolist() for elem in self.ExtractPath(self.x_best)]
            #             f.write(f"{i}\t{self.c_best}\t{self.t_best}\t{te-ts}\t{len(self.V)}\t{b_path}\t{True}\n")
            #     else:
            #         f.write(f"{i}\t{self.c_best}\t{self.t_best}\t{te-ts}\t{len(self.V)}\n")
            sol = self.sol
            
            
    
        print("self.c_best", self.c_best)
        self.path = self.ExtractPath(self.x_best)
        # print(self.path)
        plot_kino(self, i, c_best=self.c_best, tau_star=self.t_best)
        plt.pause(0.01)
        animate(self)
        self.eng.quit()
        

    def eval_arrival_time(self, x0_, x1_):
        tau_star = Matrix(self.tau_star(x0_, x1_))
        p = Poly(tau_star[0]).all_coeffs()
        time_vec = np.roots(p)
        time = max([re(t) for t in time_vec if im(t)==0 and re(t)>=0])
        return 1/time


    def eval_cost(self, x0, x1, time=None):
        if time==None:
            time = self.eval_arrival_time(x0, x1)
    
        cost = self.cost_tau(time, x, x0, x1)
        return float(cost[0][0]), float(time)

    def eval_states_and_inputs(self, x0, x1, time=None):
        if time==None:
            time = self.eval_arrival_time(x0, x1)

        states = lambdify([x, self.t], self.states(time, self.t, x, x0, x1))
        inputs = lambdify([x, self.t], Matrix(self.control_(time, self.t, x, x0, x1)))

        return states, inputs 


    def isStateFree(self, states, t_init, t_goal):
        step = 0.1

        r = np.arange(t_init/step, t_goal/step)*step

        for time_step in r:
            state = states(self.t, time_step)

            for i in range(len(self.state_limits)):
                if state[i] < self.state_limits[i][0] or state[i] > self.state_limits[i][1]:
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
        step = (t_goal-t_init)/resolution

        r = np.arange(t_init/step, t_goal/step)*step

        for time_step in r:
            inp = inputs(self.t, time_step)

            for i in range(len(self.input_limits)):
                if inp[i] < self.input_limits[i][0] or inp[i] > self.input_limits[i][1]:
                    return False

        return True


    def Sample(self):
        delta = self.delta
        node = NodeKino((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
                         np.random.uniform(-10, 10),
                         np.random.uniform(-10, 10)
                         ))

        while self.isCollision(node.node):
            node = NodeKino((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
                         np.random.uniform(-10, 10),
                         np.random.uniform(-10, 10)
                         ))
             
        return node

    def ChooseParent(self, x_rand):
        nodes, costs, times = zip(*[[np.array(node.node, dtype=np.float64), node.cost, node.time] for node in self.V])

        ret = self.eng.ChooseParent(self.matlab, np.array(nodes, dtype=np.float64), np.array(costs,dtype=np.float64), np.array(times,dtype=np.float64), np.array(x_rand.node,dtype=np.float64), self.x_goal.cost)
        ret = ret[0]

        min_cost, min_time, min_idx = ret[0], ret[1], int(ret[2])-1
 
        if min_cost == float('inf'):
            return None, None, None, None

        self.V[min_idx].children.append(x_rand)
        x_rand.cost = min_cost
        x_rand.time = min_time
        x_rand.parent = self.V[min_idx]

        return x_rand, self.V[min_idx], min_cost, min_time


    def Rewire(self, x_rand):
        nodes, costs, times, costs2goal, times2goal, nearsgoal, nodeschildren = zip(*[
                                                                    [np.array(node.node, dtype=np.float64),
                                                                     node.cost,
                                                                     node.time,
                                                                     node.cost2goal,
                                                                     node.time2goal,
                                                                     node.near_goal,
                                                                     [np.array(child.node,dtype=np.float64) for child in node.children]] for node in self.V])
                                              
        ret = self.eng.Rewire(self.matlab,
                        np.array(nodes,dtype=np.float64),
                        np.array(costs,dtype=np.float64),
                        np.array(times,dtype=np.float64),
                        np.array(costs2goal,dtype=np.float64),
                        np.array(times2goal,dtype=np.float64),
                        np.array(nearsgoal,dtype=np.float64),
                        nodeschildren,
                        np.array(x_rand.node,dtype=np.float64),
                        x_rand.cost,
                        x_rand.time,
                        self.x_goal.cost,
                        self.c_best,
                        self.t_best,
                        self.V.index(self.x_best)+1
                        )

        x_rand_children = [np.array(child,dtype=np.float64) for child in self.eng.getfield(self.matlab,"x_rand_children")]

        self.c_best, self.t_best, self.x_best = ret['c_best'], ret['t_best'], self.V[int(ret['x_best'])-1]
        
        for child in x_rand_children:
            child = child[0]
            old_parent = np.array(child[3],dtype=np.int16)-1
            idx = np.array(child[0],dtype=np.int16)-1
            self.V[old_parent].children.remove(self.V[idx])
            self.V[idx].cost = child[1]
            self.V[idx].time = child[2]
        
            x_rand.children.append(self.V[idx])
            self.V[idx].parent = self.V[-1]
        
        return x_rand
    
    def isBest(self, x_rand, i):
        cost, time = self.eval_cost(x_rand.node, self.x_goal.node)

        if x_rand.cost+cost < self.c_best:
            states, inputs = self.eval_states_and_inputs(x_rand.node, self.x_goal.node, time)
            if self.isStateFree(states, 0, time) and self.isInputFree(inputs, 0, time):
                self.c_best = x_rand.cost + cost
                self.t_best = x_rand.time + time
                self.x_goal.cost = self.c_best
                self.x_goal.time = self.t_best

                x_rand.near_goal = True
                x_rand.cost2goal = cost
                x_rand.time2goal = time

                self.x_best = x_rand
                print("TROVATA SOLUZIONE")
                if self.firstsol == None:
                    self.firstsol = len(self.V)
                self.sol +=1

                self.path = self.ExtractPath(self.x_best)
                plot_kino(self, i, c_best=self.c_best, tau_star=self.t_best)
        
        return x_rand

    def ExtractPath(self, node):
        path = [self.x_goal.node]

        while node.parent:
            # print(node.parent.x,node.parent.y)
            path.append(node.node)
            node = node.parent

        path.append(self.x_start.node)

        return path