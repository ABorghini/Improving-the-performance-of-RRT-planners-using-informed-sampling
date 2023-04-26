
from rrt_kino import *
import numpy as np
from sympy import *
from sympy.abc import x
from env_kino import NodeKino
from collections import deque
from utils import plot_kino, plot_grid, create_dir
import time as timing
from scipy.stats import multivariate_normal
import matlab.engine

from scipy.stats import norm, multivariate_normal, bernoulli
import matplotlib
# matplotlib.use('Agg')


class Informed_RRT_Star_Kino(RRT_Star_Kino):
    def __init__(
        self,
        env=None,
        x_start=[2, 2, 0, 0],
        x_goal=[98, 98, 0, 0],
        iter_max=50,
        state_dims=4,
        input_dims=2,
        state_limits=[[0, 100], [0, 100], [-10, 10], [-10, 10]],
        input_limits=[[-5, 5], [-5, 5]],
        stop_at=-np.inf,
        custom_env=False,
        seed=666,
        ppath="./simulations"
    ):

        super().__init__(
            env,
            x_start,
            x_goal,
            iter_max,
            state_dims,
            input_dims,
            state_limits,
            input_limits,
            stop_at,
            custom_env,
            seed,
            ppath
        )

        self.name = "IRRTK_star"
        self.p_best = [self.x_start.node, self.x_goal.node]  # initial best path
        plt.ion()
        custom_path = '_customEnv' if self.custom_env else '_fixedEnv'
        c_path = f'_C_{self.stop_at}' if self.stop_at!=0 else ''
        n_path = f'_N_{self.iter_max}' if self.stop_at==0 else ''
        s_path = f'_S_{self.seed}'
        self.plotting_path = f'{self.name}{n_path}{c_path}{custom_path}{s_path}'
        self.sol = 0

    def init(self):
        # create_dir(self.ppath)
        # with open(f"{self.ppath}/{self.plotting_path}.tsv", "w") as f:
        #     f.write("It\tC_best\tT_best\tTime\tN_nodi\tB_path\tSol\n")
        self.eng = matlab.engine.start_matlab()
        self.matlab = self.eng.eqs(self.state_dims, self.input_dims, self.state_limits, self.input_limits, np.array(self.env.obs_rectangle, dtype=np.float64), np.array(self.x_start.node,dtype=np.float64), np.array(self.x_goal.node,dtype=np.float64))
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

        self.x_start.cost = 0
        self.x_start.time = 0
        self.x_start.cost2goal = cost
        self.x_start.time2goal = time
        self.x_start.no_obs_cost = cost

        self.x_goal.cost = np.inf
        self.x_goal.time = np.inf
        self.x_goal.cost2goal = 0
        self.x_goal.time2goal = 0
        self.x_goal.no_obs_cost = 0

        self.V.append(self.x_start)

        # The tranistion model defines how to move from sigma_current to sigma_new
        self.transition_model = lambda x,y: [
            np.random.normal(x[i], 15, (1,))[0] if i<2 else np.random.normal(x[i], 6, (1,))[0] for i in range(len(x)) #0.5
        ]
 
        self.X_inf = [
            self.x_start,
            self.x_goal
        ]
        self.best_path = []
        self.c = 0
        self.tot = 0

        

    def planning(self):
        print("STARTING PLANNING")
        self.init()
        self.c_best = np.inf
        self.t_best = np.inf
        self.x_best = self.x_start
        sol = 0
        i = 0
        while i < self.iter_max and self.c_best > self.stop_at:
            print(f"Iteration {i} #######################################")
           
            # ts = timing.time()
            i += 1

            x_rand = self.Sample(i)

            # ChooseParent: given x_rand choose the best parent, i.e. best cost
            x_rand, min_node, min_cost, min_time = self.ChooseParent(x_rand)

            if x_rand is None:
                continue

            # Insert x_rand in the path tree
            self.V.append(x_rand)

            # Rewire: x_rand becomes parent of the nodes such that
            # the cost(x_start->x_rand->node) < cost(x_start->node)
            x_rand = self.Rewire(x_rand)

            # with open(f"{self.ppath}/{self.plotting_path}.tsv", "a") as f:
            #     if self.sol > sol:
            #         self.path = self.ExtractPath(self.x_best)
            #         plot_kino(self, i, c_best=self.c_best, tau_star=self.t_best)
            #         b_path = [elem.tolist() for elem in self.ExtractPath(self.x_best)]
            #         f.write(f"{i}\t{self.c_best}\t{self.t_best}\t{te-ts}\t{len(self.V)}\t{b_path}\t{True}\n")
            sol = self.sol

            # isBest: check whether the path cost from x_start->x_goal 
            # passing through x_rand is better than the previous best path cost 
            x_rand = self.isBest(x_rand, i) #modified for the GIF
           
            # te = timing.time()

            # with open(f"{self.ppath}/{self.plotting_path}.tsv", "a") as f:
            #     if self.sol > sol:
            #             b_path = [elem.tolist() for elem in self.ExtractPath(self.x_best)]
            #             f.write(f"{i}\t{self.c_best}\t{self.t_best}\t{te-ts}\t{len(self.V)}\t{b_path}\t{True}\n")
            #     else:
            #         f.write(f"{i}\t{self.c_best}\t{self.t_best}\t{te-ts}\t{len(self.V)}\n")
            sol = self.sol


        print("self.c_best", self.c_best)
        if self.c_best == np.inf:
            print("no solution found")
            return
        self.path = self.ExtractPath(self.x_best)
        # print(self.path)
        plot_kino(self, i, c_best=self.c_best, tau_star=self.t_best)
        plt.pause(0.01)
        # animate(self)
        self.eng.quit()

    def eval_arrival_time(self, x0_, x1_):
        tau_star = Matrix(self.tau_star(x0_, x1_))
        p = Poly(tau_star[0]).all_coeffs()
        time_vec = np.roots(p)

        time = max([re(t) for t in time_vec if im(t) == 0 and re(t) >= 0])
        # print(1/time)
        return 1 / time

    def eval_cost(self, x0, x1, time=None):

        if time == None:
            time = self.eval_arrival_time(x0, x1)

        cost = self.cost_tau(time, x, x0, x1)

        return float(cost[0][0]), float(time)

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

    def Sample(self, it):
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
            node = self.MCMC_Sampling(it)
            # print(self.c/self.tot)
        return node

    
    def isBest(self, x_rand, i):
        cost, time = self.eval_cost(x_rand.node, self.x_goal.node)

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
                x_rand.no_obs_cost = x_rand.cost + cost
                x_rand.time2goal = time
                self.x_best = x_rand 

                print("TROVATA SOLUZIONE")
                if self.firstsol == None:
                    self.firstsol = len(self.V)
                    for k in range(2,len(self.V)):
                        c1, _ = self.eval_cost(self.x_start.node,self.V[k].node)
                        c2, _ = self.eval_cost(self.V[k].node,self.x_goal.node)
                        self.V[k].no_obs_cost = c1 + c2
                    self.X_inf.extend(self.V)
                self.sol +=1

                self.path = self.ExtractPath(self.x_best)
                self.X_inf = [x for x in self.X_inf if x.no_obs_cost < self.c_best] 
                for n in self.ExtractNodes(self.x_best)[1:-1]:
                    try:
                        idx = self.X_inf.index(n)
                        if self.X_inf[idx].no_obs_cost > self.c_best:
                            self.X_inf[idx].no_obs_cost = self.c_best
                    except:
                        if n.no_obs_cost > self.c_best:
                            n.no_obs_cost = self.c_best 
                        self.X_inf.append(n)

                # self.X_inf.extend([n for n in nodes])
                # self.path = self.ExtractPath(self.x_best)

                

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


    def ExtractNodes(self, node):
        path = [self.x_goal]

        while node.parent:
            # print(node.parent.x,node.parent.y)
            path.append(node)
            node = node.parent

        path.append(self.x_start)

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

        x.no_obs_cost = cost
        self.X_inf.append(x)
        return True

    def MCMC_Sampling(self, it):  # x_i, c_best
        cont = True
        # print([x.node for x in self.X_inf])
        if self.sol == 1:
            self.last_sample = self.X_inf[np.random.randint(0, len(self.X_inf))].node
        while 1:
            # print("x_0", x_0)
            x_next = self.Metropolis_Hastings(self.last_sample, self.X_inf, it) # [np.array(n.node) for n in self.V]
            # print("x_next", x_next)

            x_next = NodeKino(x_next)
            
            if np.array_equal(np.array(x_next.node), np.array(self.last_sample)):
                # self.last_sample = self.X_inf[np.random.randint(0, len(self.X_inf))].node
                continue
            if self.in_informed(x_next):
                self.last_sample = x_next.node
                break
            else:
                self.last_sample = self.X_inf[np.random.randint(0, len(self.X_inf))].node
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


    # Computes the likelihood of the data given a sigma (new or current)
    def manual_log_like_normal(self, x, data):
        # data = the observation
        # return np.sum(-np.log(x[1] * np.sqrt(2* np.pi) )-((data-x[0])**2) / (2*x[1]**2))
        # data = [np.array(d.node,dtype=np.float64).flatten() for d in data]
        data = [np.array(node,dtype=np.float64).flatten() for node in self.path]
        # print(data)
        mean_data = np.mean(data, axis=0) # mean on the columns
        # cov = np.cov(np.array(data).T)
        # print("cov",cov)
        # print("mean", mean_data)
        cov = [[400., 0., 0., 0.], [0., 400., 0., 0.], [0., 0., 30., 0.], [0., 0., 0., 30.]]
        # cov = [[500., -150., 100., -20.], [-150., 500., -20., 100.], [300., -100., 40., 0.], [-100.,300., 0., 50.]]
        # cov = [ [ 5.00000000e+02, -1.50000000e+02,  1.00000000e+02, -2.04348934e-14],
        #         [-1.50000000e+02,  5.00000000e+02, -1.13670430e-14,  1.00000000e+02],
        #         [ 1.00000000e+02, -1.26954736e-14,  4.00000000e+01, -1.83030025e-14],
        #         [-1.22652335e-14,  1.00000000e+02, -1.78373382e-14,  5.00000000e+01]]
        # cov = [[ 5.00000000e+02, -1.50000000e+02, -3.00000000e+01, -1.05084096e-14],
        #         [-1.50000000e+02,  5.00000000e+02,  3.21120687e-15, -3.00000000e+01],
        #         [-3.00000000e+01,  6.74262167e-15,  3.00000000e+01, -2.50000000e+01],
        #         [-1.06493816e-14, -3.00000000e+01, -2.50000000e+01,  3.00000000e+01]]
       
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


    def Metropolis_Hastings(self, start, data, it):
        x = start

        r = max((15 * math.sqrt((math.log(it) / it))), 4)
        x_new = self.transition_model(x, r)
        # if random.uniform(0.0,1.0) >= 0.50:
        #     x_new = [np.float64(x[0]),np.float64(x[1]),x_new[2],x_new[3]]
        
        x_lik = self.manual_log_like_normal(x, data)
        x_new_lik = self.manual_log_like_normal(x_new, data)

        if self.acceptance(
            x_lik + np.log(self.prior(x)), x_new_lik + np.log(self.prior(x_new))
        ):
            self.c += 1
            self.tot += 1
            # print("x_new", x_new)
            return x_new

        self.tot += 1
        return x