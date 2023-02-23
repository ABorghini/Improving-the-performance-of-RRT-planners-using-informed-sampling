from re import X
# from Project.utils import plot_grid
from rrt_star import *
import numpy as np
from scipy.linalg import expm, inv
from scipy.integrate import quad
from numpy.linalg import matrix_power
from sympy import *
from sympy.abc import x
from sympy.solvers import solve
from env_kino import NodeKino, EnvKino
from random import random
from collections import deque
from utils import plot_grid, plot_kino
from jenkins_traub import *
from sympy.printing.aesaracode import aesara_function
import time as timing

class RRT_Star_Kino(RRT_Star):
    def __init__(self, env = None, x_start= None, x_goal= None, step_len= None,
                 goal_sample_rate= None, search_radius= None, iter_max= None, r_RRT= None, r_goal= None, stop_at= None):
        
        # x_start = [2,3]
        # x_goal = [1,2]

        # super().__init__(env, x_start, x_goal, step_len,
        #         goal_sample_rate, search_radius,
        #         iter_max, r_RRT, r_goal, stop_at)  

        self.name = 'RRTK*'
     
        self.state_dims = 4
        self.input_dims = 2

        self.A = zeros(self.state_dims,self.state_dims)
        self.A[0,2] = 1.0
        self.A[1,3] = 1.0
        self.B = zeros(self.state_dims, self.input_dims)
        self.B[2,0] = 1.0
        self.B[3,1] = 1.0
        self.R = eye(self.input_dims)
        self.c = zeros(self.state_dims,1)
        self.x0 = MatrixSymbol('x0',self.state_dims,1)
        self.x1 = MatrixSymbol('x1',self.state_dims,1)
        self.dist_idxs = [i for i in range(self.B.shape[0])]

        self.x_start = NodeKino([2,2,0,0])
        self.x_goal = NodeKino([50,65,0,0])

        self.state_limits = [[0, 100], [0, 100], [-10, 10], [-10, 10]]
        self.input_limits = [[-5, 5], [-5, 5]]

        self.t = Symbol("t")
        # self.t_s = Symbol("t_s")

        self.iter_max = 7
        self.delta = 0.5
        self.goal_sample_rate = 0.1
        self.V = []
        self.max_radius = 300
        self.env = env
        self.x_range = self.state_limits[0]
        self.y_range = self.state_limits[1]
        
        self.fig, self.ax = plt.subplots()
        self.delta = self.env.delta
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary


    def init(self):
        self.t = Symbol('t')
        self.ts = Symbol('ts')
        self.x = Symbol('x')
        self.x0 = MatrixSymbol('x0',self.state_dims,1)
        self.x1 = MatrixSymbol('x1',self.state_dims,1)
        self.tau = Symbol('tau')

        # Distance
        G_ = lambdify([self.tau, x], simplify(exp(self.A * (self.tau - x))) * self.B * self.R.inv() * self.B.T * simplify(exp(self.A.T * (self.tau - x))), "numpy")
        x_bar_ = lambdify([self.tau, self.ts], simplify(exp(self.A * (self.tau - self.ts))) * self.c, "numpy")
        # sympy
        self.x_bar = lambdify([self.tau, self.ts, self.x0], simplify(exp(self.A * self.tau)) * Matrix(self.x0) + integrate(Matrix(x_bar_(self.tau, self.ts)),(self.ts,0,self.tau)), "numpy")

        self.G = lambdify([self.tau, x], integrate(Matrix(G_(self.tau, x)),(x,0,self.tau)), "numpy")
        
        # sympy
        self.d = lambdify([self.x1, self.tau, self.ts, self.x0, x], Matrix(self.G(self.tau, x)).inv() * (Matrix(self.x1) - self.x_bar(self.tau, self.ts, self.x0)), "numpy")

        # Tau*
        tau_star_ = eye(1) - 2 * (self.A * Matrix(self.x1) + self.c).T * self.d(self.x1, self.tau, self.ts, self.x0, x) - self.d(self.x1, self.tau, self.ts, self.x0, x).T * self.B * self.R * self.B.T * self.d(self.x1, self.tau, self.ts, self.x0, x)
        self.tau_star = lambdify([self.x0, self.x1], tau_star_, "numpy")

        # Cost
        self.cost_tau = lambdify([self.tau, x, self.x0, self.x1], Matrix([self.tau]) + (self.x1 - self.x_bar(self.tau, self.ts, self.x0)).T * integrate(Matrix(G_(self.tau, x)),(x,0,self.tau)).inv() * (self.x1 - self.x_bar(self.tau, self.ts, self.x0)), "numpy")

        # States
        mat = Matrix([[self.A, self.B * self.R.inv() * self.B.T],
                    [zeros(self.state_dims, self.state_dims), -self.A.T]])

        exp_mat1 = lambdify([self.t, x], simplify(exp(mat * (self.t - x))), "numpy")
        exp_mat2 = lambdify([self.tau, self.t], simplify(exp(mat * (self.t - self.tau))), "numpy")

        solution_ = lambdify([self.t, x], exp_mat1(self.t, x) * Matrix([[self.c], [zeros(self.state_dims, 1)]]), "numpy")

        state0 = lambdify([self.tau, self.t, self.ts, self.x0, self.x1, x], exp_mat2(self.tau, self.t) * Matrix(BlockMatrix([[self.x1],[Matrix(self.d(self.x1, self.tau, self.ts, self.x0, x))]])), "numpy")

        sol =  lambdify([self.tau, self.t, x, self.x0, self.x1], state0(self.tau, self.t, self.ts, self.x0, self.x1, x) + integrate(Matrix(solution_(self.t, x)),(x,self.tau,self.t)), "numpy")

        self.states = lambdify([self.tau, self.t, x, self.x0, self.x1], Matrix(sol(self.tau, self.t, x, self.x0, self.x1))[0:self.state_dims], "numpy")

        # Control
        self.control_ = lambdify([self.tau, self.t, x, self.x0, self.x1], self.R.inv() * self.B.T * simplify(exp(self.A.T*(self.tau-self.t)))*integrate(Matrix(G_(self.tau, x)),(x,0,self.tau)).inv() * (Matrix(self.x1) - self.x_bar(self.tau, self.ts, self.x0)), "numpy")
        

    def planning(self):
        print('STARTING PLANNING')
        # theta, dist, x_center, C, x_best = self.init()
        self.init()
        c_best = np.inf
        t_best = np.inf
   
        x_best = self.x_start

        cost, time = self.eval_cost(self.x_start.node, self.x_goal.node, time=None)
        # cost, time = self.aux(self.x_start.node, self.x_goal.node)
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
        # otteniamo states e inputs con t, x0 e x1 fissati. poi sostituiamo t_s con i valori nel range (sampling)
        states, inputs = self.eval_states_and_inputs(self.x_start.node ,self.x_goal.node ,time) 
        print(states, inputs)
        # print("states and inputs")
        # print(states, inputs)

        i = 0
        while i<self.iter_max: #and c_best >= self.stop_at
            print(f"Iteration {i} #######################################")
            print(len(self.V))

            x_rand, min_node, min_cost, min_time  = self.get_randstate()
            stack = deque()
            stack.append((self.x_start,0,0)) #node, cost, time

            while not len(stack)==0:
                node, cost_imp, time_imp = stack.pop()

                node.cost -= cost_imp
                node.time -= time_imp
                if node.near_goal:
                    if node.cost + node.cost2goal < c_best:
                        c_best = node.cost + node.cost2goal
                        t_best = node.time + node.time2goal
                        x_best = node
                    continue
                    
                
                diff = cost_imp
                time_diff = time_imp
                   
                # per ogni nodo abbiamo il costo start-nodo
                # costo start-x_i (c'è) + x_i-nodo (da calcolare)
                # x_i diventa padre di un nodo ottimizzando il costo del path
                partial_cost, partial_time = self.eval_cost(x_rand.node, node.node, time=None)
                partial_cost = partial_cost[0]
                if partial_cost < self.max_radius: # and self.is_collision()

                    new_cost = partial_cost + x_rand.cost
    
                    if new_cost < node.cost:
                        states, inputs = self.eval_states_and_inputs(x_rand.node ,node.node , partial_time) 
                        if self.is_state_free(states, 0, partial_time) and self.is_input_free(inputs, 0, partial_time):
                            
                            new_time = partial_time + x_rand.time
                            diff = node.cost - new_cost
                            time_diff = node.time - new_time

                            self.V[self.V.index(node)].parent = x_rand
                            self.V[self.V.index(node)].cost = new_cost
                            x_rand.children.append(node)
                            
                            self.V[self.V.index(node)].time = new_time

                
                for child in node.children:
                    stack.append((child, diff, time_diff))       


            cost, time = self.eval_cost(x_rand.node, self.x_goal.node)
            cost = cost[0]
            if x_rand.cost+cost < c_best:
                states, inputs = self.eval_states_and_inputs(x_rand.node, self.x_goal.node, time)
                if self.is_state_free(states, 0, time) and self.is_input_free(inputs, 0, time):
                    c_best = x_rand.cost + cost
                    t_best = x_rand.time + time
                    self.x_goal.cost = c_best
                    self.x_goal.time = t_best
                    x_rand.near_goal = True
                    x_rand.cost2goal = cost
                    # x_rand.time2goal = time
                    x_best = x_rand
                    print("TROVATA SOLUZIONE")

            self.V.append(x_rand)

            # print("iter",i)
            # if i % 20 == 0 or i == self.iter_max-1:
            #     # print("iter", i)
            #     plot(self, i)
            i+=1

        print("c_best", c_best)
        self.path = self.ExtractPath(x_best)
        print(self.path)
        plot_kino(self)
        # print("path", self.path)
        # plot(self, 1)
        # plt.plot([x for x, _ in self.path], [y for _, y in self.path], color=(1.0,0.0,0.0,1.0))
        # plt.pause(0.001)
        # plt.show()


    # def states_eq(self, tau_star, x0, x1):

    #     return self.states.subs({self.tau: tau_star, self.x0: x0, self.x1: x1}).as_explicit()
    

    # def control(self, tau, x0, x1):

    #     return self.control_.subs({self.tau: tau, self.x0: x0, self.x1: x1}).as_explicit()
    
    # to be continued###################################################
    def sq_distance(self, t, t_s, x0, x1):
        # print('SQ DISTANCE')
        m = ((x1[self.dist_idxs] - self.states_eq(t, t_s, x0, x1)[self.dist_idxs])**2)
        res = 0
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                res += m[i,j]
        
        res_distance = res - self.min_dist**2

        #print(res_distance)
        #print('res_distance')

        return res_distance


    # def cost_eq(self, time, x0, x1):

    #     return self.cost_tau.subs({self.tau: time, self.x0: x0, self.x1: x1}).as_explicit()


    def eval_arrival_time(self, x0_, x1_):
        # tau_star = self.tau_star.subs({self.x0: x0_, self.x1: x1_}).as_explicit()
        # s = timing.time()
        tau_star = Matrix(self.tau_star(x0_, x1_))
        # e = timing.time()
        # print("lamb", e-s)
        
        p = Poly(tau_star[0]).all_coeffs()
        time_vec = np.roots(p)

        time = max([re(t) for t in time_vec if im(t)==0 and re(t)>=0])
        return 1/time


    def eval_cost(self, x0, x1, time=None):
        # print('EVAL COST')
        # x0 = x0
        # x1 = x1

        if time==None:
            # s = timing.time()
            time = self.eval_arrival_time(x0, x1)
            # e = timing.time()
            # print("tau star", e-s)

        cost = self.cost_tau(time, x, x0, x1)
        # print('cost and time')
        # print(cost,time)
        return cost, time

    def eval_states_and_inputs(self, x0, x1, time=None):
        # print('EVAL STATES AND INPUTS')
        # x0 = x0
        # x1 = x1

        if time==None:
            time = self.eval_arrival_time(x0, x1)

        # s = timing.time()
        states = lambdify([x, self.t], self.states(time, self.t, x, x0, x1))
        # e = timing.time()
        # print("states", e-s)
        inputs = lambdify([x, self.t], Matrix(self.control_(time, self.t, x, x0, x1)))
        # e2 = timing.time()
        # print("inputs", e2-e)

        return states, inputs 


    def is_state_free(self, states, t_init, t_goal):
        step = 1.0

        r = np.arange(t_init/step, t_goal/step)*step

        for time_step in r:
            # s = timing.time()
            state = states(self.t, time_step)
            # e = timing.time()
            # print("state internal", e-s)

            # print("state")
            # print(state)
            for i in range(len(self.state_limits)):
                if state[i] < self.state_limits[i][0] or state[i] > self.state_limits[i][1]:
                    # print("OUT")
                    return False

            if self.is_collision(state):
                # print("COLLISION")
                return False

        return True


    def is_collision(self, state):
        if self.env.is_inside_obs(state):
            return True
        return False
        
        # for ob in self.env.
        # closest = min(max(state.extract([0,1],[0,-1]),))

 
    def is_input_free(self, inputs, t_init, t_goal):
        resolution = 20
        step = (t_goal-t_init)/resolution

        r = np.arange(t_init/step, t_goal/step)*step

        for time_step in r:
            # s = timing.time()
            inp = inputs(self.t, time_step)
            # e = timing.time()
            # print("input internal", e-s)

            for i in range(len(self.input_limits)):
                if inp[i] < self.input_limits[i][0] or inp[i] > self.input_limits[i][1]:
                    return False

        return True


    def SampleFromSpace(self):
        delta = self.delta
        node = NodeKino((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
                         np.random.uniform(-10, 10),
                         np.random.uniform(-10, 10)
                         ))

        while self.is_collision(node.node):
            node = NodeKino((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta),
                         np.random.uniform(-10, 10),
                         np.random.uniform(-10, 10)
                         ))
             
        return node

    def get_randstate(self):

        sample_ok = False
        min_node = None

        while not sample_ok:

            x_rand = self.SampleFromSpace()
            print("x_rand", x_rand.node)
            min_cost = np.inf
            min_time = np.inf

            for node in self.V:
                # s = timing.time()
                cost, time = self.eval_cost(node.node, x_rand.node)
                cost = cost[0]
                # cost, time = self.aux(node.node, x_rand.node)
                # print("node.cost", node.node, node.cost, cost)
                # print("cost goal", self.x_goal.cost)
                if cost < self.max_radius and node.cost+cost < min_cost and node.cost+cost < self.x_goal.cost:

                    states, inputs = self.eval_states_and_inputs(node.node, x_rand.node, time)
                    # print("states")
                    # print(states)
                    # print("inputs")
                    # print(inputs)
                    if self.is_state_free(states, 0, time) and self.is_input_free(inputs, 0, time):
                        print("EUREKA")
                        sample_ok = True
                        min_cost = cost+node.cost
                        min_time = time+node.time
                        min_node = node

                # e = timing.time()
                # print(e-s)

        # self.V[self.V.index(min_node)].cost = min_cost
        # self.V[self.V.index(min_node)].time = min_time
        self.V[self.V.index(min_node)].children.append(x_rand)
        x_rand.cost = min_cost
        x_rand.time = min_time
        x_rand.parent = self.V[self.V.index(min_node)]
        # self.V.append(x_rand)

        return x_rand, min_node, min_cost, min_time


    def ExtractPath(self, node):
        path = [self.x_goal.node]

        while node.parent:
            # print(node.parent.x,node.parent.y)
            path.append(node.node)
            node = node.parent

        path.append(self.x_start.node)

        return path


    @staticmethod
    def Line(x_start, x_goal):
        return math.hypot(x_goal.x - x_start.x, x_goal.y - x_start.y)


    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def create_env(env, rnd, n_obs = 0):
    if rnd:
        while len(env.obs_rectangle)!=n_obs:
            rnd_x = random.uniform(env.x_range[0], env.x_range[1])
            rnd_y = random.uniform(env.y_range[0], env.y_range[1])
            rnd_w = random.uniform(0.5, 1.5)
            rnd_h = random.uniform(0.5, 1.5)
            
        env.add_rectangle(rnd_x,rnd_y,rnd_w,rnd_h)
        #print("len",len(env.obs_rectangle))
        #print(rnd_x,rnd_y,rnd_w,rnd_h)

        print("Environment done!")

    else: #fixed environment
        obs_rectangle = [
                [60,0,10,20],
                [60,30,10,70],
                [30,0,10,70],
                [30,80,10,20]
            ]

        for rect in obs_rectangle:
            env.add_rectangle(rect[0], rect[1], rect[2], rect[3])


def main():
    # A = zeros(4,4)
    # B = eye(4)
    # A[0,2] = 1.0
    # A[1,3] = 1.0
    # a = A*B*x
    # print(a.integrate((x,0,3)))
    x_start = [2,2,0,0]
    x_goal = [98,98,0,0]
    env = EnvKino(x_start=x_start, x_goal=x_goal, delta=0.5)
    create_env(env, rnd=False)
    print("env", type(env))
    rrtkino = RRT_Star_Kino(env = env)
    # plot_grid(rrtkino.name, rrtkino)
    # plt.show()
    # print(rrtkino.aux(rrtkino.x_start.node,rrtkino.x_goal.node))
    rrtkino.planning()

    # t = Symbol('t')
    # t_s = Symbol('t_s')

    # f = rrtkino.tau_star(0, t_s)
    # print(f)
    # print(solve(f, t_s))
    # t = Symbol('t')
    # p1 = S.One
    # it = 0

    # while it < 20 and len(p1.args) <= 1:
    #     p1 = simplify(rrtkino.tau_star(0, 3) * t**it)
    #     print(Matrix(p1.as_coeff_mul(t)[0]))
    #     m = Matrix(p1.as_coeff_mul(t)[0])

    #     p1 = p1.as_coeff_mul()[0]
    #     it += 1
    #     break

    # print(m.T)
    # A = zeros(4,4)
    # A[1,2] = 1
    # A[3,2] = 2
    # A[0,2] = 1
    # A = simplify(A*t)
    # print(A)
    # print(A.as_coeff_mul(t))
    


if __name__ == '__main__':
    main()
