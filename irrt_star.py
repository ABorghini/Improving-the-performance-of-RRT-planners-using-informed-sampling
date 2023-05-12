import os
import sys
import math
from turtle import width
import numpy as np
import matplotlib.pyplot as plt
from rrt_star import RRT_Star
import random
from env import Node, Env
from utils import plot, animate
import time as timing
from scipy.stats import multivariate_normal
# import matplotlib
# matplotlib.use('Agg')

class Informed_RRT_Star(RRT_Star):
    def __init__(self, env, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max, r_RRT, fixed_near_radius, r_goal, stop_at, rnd, n_obs, custom_env, seed, env_seed, mh=False):
        
        super().__init__(env, x_start, x_goal, step_len,
                        goal_sample_rate, search_radius,
                        iter_max, r_RRT, fixed_near_radius, r_goal, stop_at, 
                        rnd, n_obs, custom_env, seed, env_seed)   
        self.name = 'IRRT_star'
        self.X_soln = set()
        self.mh = mh
        if self.stop_at!=0:
            self.iter_max = 10000

        rnd_path = 'randEnv' if rnd else 'customEnv' if self.custom_env else 'fixedEnv'
        c_path = f'_C_{self.stop_at}_' if self.stop_at!=0 else ''
        n_path = f'_N_{self.iter_max}_' if self.stop_at==0 else ''
        s_path = f'_ES_{self.env_seed}_S_{self.seed}'
        o_path = f'_O_{n_obs}' if rnd else ''
        mh_path = f'_MH_' if self.mh else ''

        self.plotting_path = f'{self.name}{n_path}{c_path}{rnd_path}{s_path}{o_path}{mh_path}'

        self.ppath = 'simulations'

        # with open(f"{self.ppath}/{self.name}_nodes.tsv", "w") as f:
        #     f.write("seed\tx_start\tx_goal\tb_path\n")

        with open(f"{self.ppath}/{self.plotting_path}.tsv", "w") as f:
            f.write("It\tC_best\tTime\tN_nodi\tB_path\tSol\n")

        self.duration = 0 #to add for the graphic analysis without the plots
        # The tranistion model defines how to move from sigma_current to sigma_new
        if self.mh:
            tr = 15
            if self.rnd:
                tr = 3
            
            self.transition_model = lambda k: [
                np.random.normal(k[0], tr, (1,))[0],
                np.random.normal(k[1], tr, (1,))[0] #0.5
            ]

            self.X_inf = [self.x_start,self.x_goal]
            self.x_start.no_obs_cost = self.Line(self.x_start,self.x_goal)
            self.x_goal.no_obs_cost = 0
            self.c = 0
            self.tot = 0


    def planning(self):
        theta, dist, x_center, C, x_best = self.init()
        c_best = np.inf
   
        x_best = self.x_start
        self.sol = 0
        i = 0
        while i<self.iter_max and c_best > self.stop_at:
            print(f"Iteration {i} #######################################")
            ts = timing.time()
            trovata = False
            aggiunto = False


            if self.X_soln:
                cost = {node: self.Cost(node, self.x_goal) for node in self.X_soln}
                x_best = min(cost, key=cost.get) 
                if cost[x_best] != c_best:
                    if self.sol != 1:
                        self.sol += 1
                    trovata = True
                    self.path = self.ExtractNodes(x_best)
                
                c_best = cost[x_best]
                if c_best == self.stop_at:
                    break

            x_rand = self.Sample(c_best, dist, x_center, C)
            x_nearest = self.Nearest(x_rand)
            x_new = self.Steer(x_nearest, x_rand)

            if not self.env.isCollision(x_nearest, x_new):
                X_near = self.Near(self.V, x_new, self.r_RRT, self.fixed_near_radius) # r_RRT
                c_min = self.Cost(x_nearest, x_new)

                # choose parent
                x_new, _ = self.ChooseParent(X_near, x_new, c_min)

                self.V.append(x_new)
                aggiunto = True

                # Rewire
                self.Rewire(X_near, x_new)
                
                x_new = self.V[self.V.index(x_new)]
               
                if self.InGoalRegion(x_new):
                    if not self.env.isCollision(x_new, self.x_goal):
                        if self.sol == 0:
                            c_ = self.Cost(x_new)
                            if c_ < c_best:
                                x_best = x_new
                                c_best = c_
                                self.path = self.ExtractNodes(x_best)
                                self.sol += 1
                                if self.mh:
                                    for node in self.V:
                                        node.no_obs_cost = self.Line(self.x_start,node)+self.Line(node,self.x_goal)
                                        if node.no_obs_cost < c_best:
                                            self.X_inf.append(node)    
                                    
                        if self.mh:
                            self.X_inf = [x for x in self.X_inf if x.no_obs_cost < c_best] 
                            for n in self.path[1:-1]:
                                try:
                                    idx = self.X_inf.index(n)
                                    if self.X_inf[idx].no_obs_cost > c_best:
                                        self.X_inf[idx].no_obs_cost = c_best
                                except:
                                    if n.no_obs_cost > c_best:
                                        n.no_obs_cost = c_best 
                                    self.X_inf.append(n)
                            self.mean_data = np.mean([np.array([element.x,element.y],dtype=np.float64) for element in self.path],axis=0)
                            # self.mean_data = np.mean([np.array([element.x,element.y],dtype=np.float64) for element in self.X_inf],axis=0)
                            # print(self.mean_data)
                        self.X_soln.add(x_new)
                        
            te = timing.time()
            if aggiunto:
                with open(f"{self.ppath}/{self.plotting_path}.tsv", "a") as f:
                    b_path = self.ExtractPath(x_best)
                    if trovata:
                        f.write(f"{i}\t{c_best}\t{te-ts}\t{len(self.V)}\t{b_path}\t{trovata}\n")
                    else:
                        f.write(f"{i}\t{c_best}\t{te-ts}\t{len(self.V)}\n")
                        
            if i % 20 == 0 or i == self.iter_max-1:
                plot(self, i, x_center=x_center, c_best=c_best, dist=dist, theta=theta)
            
            i+=1


        # self.path = self.ExtractPath(x_best)
        plot(self, i, x_center=x_center, c_best=c_best, dist=dist, theta=theta)
        # plt.plot([x for x, _ in self.path], [y for _, y in self.path], color=(1.0,0.0,0.0,1))
        plt.savefig(f'{self.plotting_path}/img_{i}1')
        plt.pause(0.001)
        plt.show()
        animate(self)


    def Near(self, V, x_new, search_radius = 20, fixed=False):
        n = len(V) + 1
        # r2 = (search_radius * math.sqrt((math.log(n) / n))) ** 2
        if fixed:
            r = search_radius
        else:
            rnear = 20
            if self.rnd:
                rnear = 3
            r = min((search_radius * math.sqrt((math.log(n) / n))), rnear)
            # print(r)
        # self.step_len = r
        r2 = r**2
        dist_table = [(n.x - x_new.x) ** 2 + (n.y - x_new.y) ** 2 for n in V]
        X_near = [v for v in V if dist_table[V.index(v)] <= r2 and not self.env.isCollision(v, x_new)]
        
        return X_near

    def Sample(self, c_max, c_min, x_center, C):
        if c_max < np.inf: #at least a solution has been found
            if self.mh:
                x_rand = self.MCMC_Sampling(c_max)
                # print(self.c/self.tot)
            else:
                #radii of the ellipsoid
                r = [c_max / 2.0,
                    math.sqrt(c_max ** 2 - c_min ** 2) / 2.0] 
        
                L = np.diag(r)
                x_rand = None

                while not self.inside_boundaries(x_rand):
                    x_ball = self.SampleUnitBall()
                    x_rand = np.dot(np.dot(C, L), x_ball) + x_center

                x_rand = Node((x_rand[(0, 0)], x_rand[(1, 0)]))
        else:
            x_rand = self.SampleFromSpace()
        
        return x_rand


    def InGoalRegion(self, node):
        if self.Line(node, self.x_goal) < self.r_goal:
            return True

        return False

    def MCMC_Sampling(self, c_best):  # x_i, c_best
        if self.sol == 1:
            self.last_sample = self.X_inf[np.random.randint(0, len(self.X_inf))]
        while 1:
            x_next = self.Metropolis_Hastings([self.last_sample.x,self.last_sample.y]) # [np.array(n.node) for n in self.V]
            x_next = Node(x_next)
            if np.array_equal(np.array(x_next), np.array(self.last_sample)):
                self.last_sample = self.X_inf[np.random.randint(0, len(self.X_inf))]
                continue
            if self.in_informed(x_next, c_best):
                self.last_sample = x_next
                break
            else:
                self.last_sample = self.X_inf[np.random.randint(0, len(self.X_inf))]
                continue

        return x_next

    def in_informed(self, x, c_best):
        # check collision
        if self.env.is_inside_obs(x):
            #print('Collide')
            return False

        # compute cost: x_start -> x -> x_goal
        cost1 = self.Line(self.x_start, x)
        
        cost2 = self.Line(x, self.x_goal)

        cost = cost1 + cost2
        if cost >= c_best:
            #print('non migliora')
            return False

        x.no_obs_cost = cost
        self.X_inf.append(x)
        # self.mean_data = np.mean([np.array([element.x,element.y],dtype=np.float64) for element in self.X_inf],axis=0)
        # print(self.mean_data)
        return True
    
    def prior(self, x):
        # returns 1 for all valid values of the sample. Log(1) =0, so it does not affect the summation.
        # returns 0 for all invalid values of the sample (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
        # It makes the new sample infinitely unlikely.

        if self.env.x_range[0] < x[0] < self.env.x_range[1] and \
            self.env.y_range[0] < x[1] < self.env.y_range[1]:
            return 1
        return 0
    
    # Computes the likelihood of the data given a sigma (new or current)
    def manual_log_like_normal(self, x):
        # data = the observation
        # return np.sum(-np.log(x[1] * np.sqrt(2* np.pi) )-((data-x[0])**2) / (2*x[1]**2))
        # mean_data = np.mean(data, axis=0) # mean on the columns
        cov = [[600., 0.], [-150., 600.]] #base value
        if self.rnd:
            cov = [[7., 4.], [4., 7.]] #base value
        return np.log(multivariate_normal.pdf(
            np.array(x).flatten(), mean=self.mean_data.flatten(), cov=cov
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
    
    def Metropolis_Hastings(self, start):
        x = start
       
        x_new = self.transition_model(x)
        x_lik = self.manual_log_like_normal(x)
        x_new_lik = self.manual_log_like_normal(x_new)

        if self.acceptance(
            x_lik + np.log(self.prior(x)), x_new_lik + np.log(self.prior(x_new))
        ):
            # self.c += 1
            # self.tot += 1
            return x_new
        # self.tot += 1
        return x

    def ExtractNodes(self, node):
        path = [self.x_goal]

        while node.parent:
            # print(node.parent.x,node.parent.y)
            path.append(node)
            node = node.parent

        path.append(self.x_start)

        return path

    @staticmethod
    def SampleUnitBall():
        while True:
            x, y = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y]])
