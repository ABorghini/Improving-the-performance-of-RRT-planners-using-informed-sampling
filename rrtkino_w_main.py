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
import random
from collections import deque
from utils import animate, plot_kino
from jenkins_traub import *
from sympy.printing.aesaracode import aesara_function
import time as timing

SEED = 666
random.seed(SEED)
np.random.seed(SEED)

# np.random.seed(0)

class RRT_Star_Kino(RRT_Star):
    def __init__(self, env = None, x_start = [2,2,0,0], x_goal = [50,65,0,0], max_radius = 100,
                 iter_max = 50, state_dims = 4, input_dims = 2, state_limits = [[0, 100], [0, 100], [-10, 10], [-10, 10]],
                 input_limits = [[-5, 5], [-5, 5]], stop_at= None):
        
        # x_start = [2,3]
        # x_goal = [1,2]

        # super().__init__(env, x_start, x_goal, step_len,
        #         goal_sample_rate, search_radius,
        #         iter_max, r_RRT, r_goal, stop_at)  

        self.name = 'RRTK_star'
        plt.ion()
        self.state_dims = state_dims
        self.input_dims = input_dims

        self.A = zeros(self.state_dims,self.state_dims)
        self.A[0,2] = 1.0
        self.A[1,3] = 1.0
        self.B = zeros(self.state_dims, self.input_dims)
        self.B[2,0] = 1.0
        self.B[3,1] = 1.0
        self.R = eye(self.input_dims)
        self.c = zeros(self.state_dims,1)
        # self.dist_idxs = [i for i in range(self.B.shape[0])]
        self.x_start = NodeKino(x_start)
        self.x_goal = NodeKino(x_goal)

        self.state_limits = state_limits
        self.input_limits = input_limits

        # self.t = Symbol("t")
        # self.t_s = Symbol("t_s")

        self.iter_max = iter_max
        # self.goal_sample_rate = 0.1
        self.V = []
        self.max_radius = max_radius
        self.env = env
        self.x_range = self.state_limits[0]
        self.y_range = self.state_limits[1]
        
        self.fig, self.ax = plt.subplots()
        self.delta = self.env.delta
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        n_path = f'_N_{self.iter_max}'
        self.plotting_path = f'{self.name}{n_path}'
        self.sol = 0


    def init(self):
        self.t = Symbol('t')
        self.ts = Symbol('ts')
        self.x = Symbol('x')
        self.x0 = MatrixSymbol('x0',self.state_dims,1)
        self.x1 = MatrixSymbol('x1',self.state_dims,1)
        self.tau = Symbol('tau')

        # Distance
        # G_ = lambdify([self.tau, x], simplify(exp(self.A * (self.tau - x))) * self.B * self.R.inv() * self.B.T * simplify(exp(self.A.T * (self.tau - x))), "numpy")
        # x_bar_ = lambdify([self.tau, self.ts], simplify(exp(self.A * (self.tau - self.ts))) * self.c, "numpy")
        self.x_bar = lambdify([self.tau, self.ts, self.x0], simplify(exp(self.A * self.tau)) * Matrix(self.x0) + integrate(simplify(exp(self.A * (self.tau - self.ts))) * self.c, (self.ts,0,self.tau)), "numpy")
        self.G = lambdify([self.tau, x], integrate(simplify(exp(self.A * (self.tau - x))) * self.B * self.R.inv() * self.B.T * simplify(exp(self.A.T * (self.tau - x))),(x,0,self.tau)), "numpy")
        self.d = lambdify([self.x1, self.tau, self.ts, self.x0, x], Matrix(self.G(self.tau, x)).inv() * (Matrix(self.x1) - self.x_bar(self.tau, self.ts, self.x0)), "numpy")
        
        # Tau*
        tau_star_ = eye(1) - 2 * (self.A * Matrix(self.x1) + self.c).T * self.d(self.x1, self.tau, self.ts, self.x0, x) - self.d(self.x1, self.tau, self.ts, self.x0, x).T * self.B * self.R * self.B.T * self.d(self.x1, self.tau, self.ts, self.x0, x)
        self.tau_star = lambdify([self.x0, self.x1], tau_star_, "numpy")

        # Cost
        # self.cost_tau = lambdify([self.tau, x, self.x0, self.x1], Matrix([self.tau]) + (self.x1 - self.x_bar(self.tau, self.ts, self.x0)).T * integrate(Matrix(G_(self.tau, x)),(x,0,self.tau)).inv() * (self.x1 - self.x_bar(self.tau, self.ts, self.x0)), "numpy")
        self.cost_tau = lambdify([self.tau, x, self.x0, self.x1], Matrix([self.tau]) + (self.x1 - self.x_bar(self.tau, self.ts, self.x0)).T * Matrix(self.G(self.tau, x)).inv() * (self.x1 - self.x_bar(self.tau, self.ts, self.x0)), "numpy")

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
        # self.control_ = lambdify([self.tau, self.t, x, self.x0, self.x1], self.R.inv() * self.B.T * simplify(exp(self.A.T*(self.tau-self.t)))*integrate(Matrix(G_(self.tau, x)),(x,0,self.tau)).inv() * (Matrix(self.x1) - self.x_bar(self.tau, self.ts, self.x0)), "numpy")
        self.control_ = lambdify([self.tau, self.t, x, self.x0, self.x1], self.R.inv() * self.B.T * simplify(exp(self.A.T*(self.tau-self.t)))*Matrix(self.G(self.tau, x)).inv() * (Matrix(self.x1) - self.x_bar(self.tau, self.ts, self.x0)), "numpy")
        
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

    def planning(self):
        print('STARTING PLANNING')
        # theta, dist, x_center, C, self.x_best = self.init()
        self.init()
        self.c_best = np.inf
        self.t_best = np.inf
   
        self.x_best = self.x_start

        # otteniamo states e inputs con t, x0 e x1 fissati. poi sostituiamo t_s con i valori nel range (sampling)
        # states, inputs = self.eval_states_and_inputs(self.x_start.node ,self.x_goal.node ,time) 
        # print(states, inputs)
        # print("states and inputs")
        # print(states, inputs)

        i = 0
        while i<self.iter_max: #and self.c_best >= self.stop_at
            print(f"Iteration {i} #######################################")
            print(len(self.V))
            i += 1
            # x_rand, min_node, min_cost, min_time  = self.get_randstate()

            x_rand = self.Sample()
            x_rand, min_node, min_cost, min_time  = self.ChooseParent(x_rand)

            if x_rand is None:
                continue

            x_rand = self.Rewire(x_rand)       

            x_rand =  self.isBest(x_rand, i)
            
            self.V.append(x_rand)


        print("self.c_best", self.c_best)
        self.path = self.ExtractPath(self.x_best)
        print(self.path)
        plot_kino(self, i, c_best=self.c_best[0], tau_star=self.t_best)
        animate(self)
        

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

    def eval_arrival_time(self, x0_, x1_):
        # tau_star = self.tau_star.subs({self.x0: x0_, self.x1: x1_}).as_explicit()
        # s = timing.time()
        tau_star = Matrix(self.tau_star(x0_, x1_))

        # e = timing.time()
        # print("lamb", e-s)
        
        p = Poly(tau_star[0]).all_coeffs()
        time_vec = np.roots(p)

        time = max([re(t) for t in time_vec if im(t)==0 and re(t)>=0])
        #print(1/time)
        return 1/time


    def eval_cost(self, x0, x1, time=None):
        # print('EVAL COST')
        # x0 = x0
        # x1 = x1

        if time==None:
            s = timing.time()
            time = self.eval_arrival_time(x0, x1)
            e = timing.time()
            #print("tau star", e-s)

        s = timing.time()
        cost = self.cost_tau(time, x, x0, x1)
        e = timing.time()
        #print("cost", e-s)
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


    def isStateFree(self, states, t_init, t_goal):
        step = 0.3

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

            if self.isCollision(state):
                # print("COLLISION")
                return False

        return True


    def isCollision(self, state):
        if self.env.is_inside_obs(state):
            return True
        return False
        
        # for ob in self.env.
        # closest = min(max(state.extract([0,1],[0,-1]),))

 
    def isInputFree(self, inputs, t_init, t_goal):
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

        # sample_ok = False
        min_node = None

        # while not sample_ok:

            # x_rand = self.Sample()
        #print("x_rand", x_rand.node)
        min_cost = np.inf
        min_time = np.inf

        for node in self.V:
            s = timing.time()
            cost, time = self.eval_cost(node.node, x_rand.node)
            e = timing.time()
            #print(e-s)
            s = timing.time()
            xbar = self.x_bar(time, self.ts, node.node)
            e = timing.time()
            #print(e-s, xbar)
            cost = cost[0]

            # se eval cost impiega 0.0x, xbar 0.000x

            # cost, time = self.aux(node.node, x_rand.node)
            # print("node.cost", node.node, node.cost, cost)
            # print("cost goal", self.x_goal.cost)
            if cost < self.max_radius and node.cost+cost < min_cost and node.cost+cost < self.x_goal.cost:

                states, inputs = self.eval_states_and_inputs(node.node, x_rand.node, time)
                # print("states")
                # print(states)
                # print("inputs")
                # print(inputs)
                if self.isStateFree(states, 0, time) and self.isInputFree(inputs, 0, time):
                    print("EUREKA")
                    sample_ok = True
                    min_cost = cost+node.cost
                    min_time = time+node.time
                    min_node = node

                # e = timing.time()
                # print(e-s)
        
        if min_node is None:
            return None, None, None, None

        # self.V[self.V.index(min_node)].cost = min_cost
        # self.V[self.V.index(min_node)].time = min_time
        self.V[self.V.index(min_node)].children.append(x_rand)
        x_rand.cost = min_cost
        x_rand.time = min_time
        x_rand.parent = self.V[self.V.index(min_node)]
        # self.V.append(x_rand)

        return x_rand, min_node, min_cost, min_time

    def Rewire(self, x_rand):
        stack = deque()
        stack.append((self.x_start,0,0)) #node, cost, time

        while not len(stack)==0:
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
            partial_cost, partial_time = self.eval_cost(x_rand.node, node.node, time=None)
            partial_cost = partial_cost[0]
            if partial_cost < self.max_radius: # and self.isCollision()

                new_cost = partial_cost + x_rand.cost

                if new_cost < node.cost:
                    states, inputs = self.eval_states_and_inputs(x_rand.node ,node.node , partial_time) 
                    if self.isStateFree(states, 0, partial_time) and self.isInputFree(inputs, 0, partial_time):
                        
                        new_time = partial_time + x_rand.time
                        diff = node.cost - new_cost
                        time_diff = node.time - new_time

                        self.V[self.V.index(node)].parent = x_rand
                        self.V[self.V.index(node)].cost = new_cost
                        x_rand.children.append(node)
                        
                        self.V[self.V.index(node)].time = new_time

            
            for child in node.children:
                stack.append((child, diff, time_diff))
            
        return x_rand
    
    def isBest(self, x_rand, i):
        cost, time = self.eval_cost(x_rand.node, self.x_goal.node)
        cost = cost[0]
        if x_rand.cost+cost < self.c_best:
            states, inputs = self.eval_states_and_inputs(x_rand.node, self.x_goal.node, time)
            if self.isStateFree(states, 0, time) and self.isInputFree(inputs, 0, time):
                self.c_best = x_rand.cost + cost
                self.t_best = x_rand.time + time
                self.x_goal.cost = self.c_best
                self.x_goal.time = self.t_best
                x_rand.near_goal = True
                x_rand.cost2goal = cost
                # x_rand.time2goal = time
                self.x_best = x_rand
                print("TROVATA SOLUZIONE")

                self.sol +=1
                self.path = self.ExtractPath(self.x_best)
                print(self.path)

                print('c_best format:', self.c_best)
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

def create_env(env, rnd, n_obs = 0):
    if rnd:
        while len(env.obs_rectangle)!=n_obs:
            rnd_x = random.random.uniform(env.x_range[0], env.x_range[1])
            rnd_y = random.random.uniform(env.y_range[0], env.y_range[1])
            rnd_w = random.random.uniform(0.5, 1.5)
            rnd_h = random.random.uniform(0.5, 1.5)
            
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
    #print("env", type(env))
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
