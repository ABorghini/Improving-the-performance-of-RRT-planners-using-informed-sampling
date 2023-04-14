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

        self.plotting_path = f'{self.name}{n_path}{c_path}{rnd_path}{s_path}{o_path}'

        self.duration = 0 #to add for the graphic analysis without the plots
        # The tranistion model defines how to move from sigma_current to sigma_new
        if self.mh:

            self.transition_model = lambda k,z: [
                np.random.normal(k[0], z*1, (1,))[0],
                np.random.normal(k[1], z*1, (1,))[0] #0.5
            ]

            self.X_inf = []


    def planning(self):
        theta, dist, x_center, C, x_best = self.init()
        c_best = np.inf
   
        x_best = self.x_start
        sol = 0
        i = 0
        while i<self.iter_max and c_best > self.stop_at:
            if self.X_soln:
                cost = {node: self.Cost(node, self.x_goal) for node in self.X_soln}
                x_best = min(cost, key=cost.get)
                c_best = cost[x_best]
                if c_best == self.stop_at:
                    break

            x_rand = self.Sample(c_best, dist, x_center, C, self.mh, i)
            x_nearest = self.Nearest(x_rand)
            x_new = self.Steer(x_nearest, x_rand)

            if not self.env.isCollision(x_nearest, x_new):
                X_near = self.Near(self.V, x_new, self.r_RRT, self.fixed_near_radius) # r_RRT
                c_min = self.Cost(x_nearest, x_new)

                # choose parent
                x_new, _ = self.ChooseParent(X_near, x_new, c_min)

                self.V.append(x_new)

                # Rewire
                self.Rewire(X_near, x_new)
                
                x_new = self.V[self.V.index(x_new)]
               
                if self.InGoalRegion(x_new):
                    if not self.env.isCollision(x_new, self.x_goal):
                        if self.mh:
                            self.X_inf.extend([element for element in self.ExtractPath(x_new) if element not in self.X_inf])
                        self.X_soln.add(x_new)
                        sol += 1

            if i % 20 == 0 or i == self.iter_max-1:
                plot(self, i, x_center=x_center, c_best=c_best, dist=dist, theta=theta)
            
            i+=1


        self.path = self.ExtractPath(x_best)
        plot(self, i, x_center=x_center, c_best=c_best, dist=dist, theta=theta)
        plt.plot([x for x, _ in self.path], [y for _, y in self.path], color=(1.0,0.0,0.0,1))
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
            r = max((search_radius * math.sqrt((math.log(n) / n))), self.step_len)
        # print(r)
        # self.step_len = r
        r2 = r**2
        dist_table = [(n.x - x_new.x) ** 2 + (n.y - x_new.y) ** 2 for n in V]
        X_near = [v for v in V if dist_table[V.index(v)] <= r2 and not self.env.isCollision(v, x_new)]
        
        return X_near

    def Sample(self, c_max, c_min, x_center, C, mh=False, it=0):
        if c_max < np.inf: #at least a solution has been found
            if mh:
                x_rand = self.MCMC_Sampling(it, c_max)
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

    def MCMC_Sampling(self, it, c_best):  # x_i, c_best
        cont = True
        while 1:
            x_0 = self.X_inf[np.random.randint(0, len(self.X_inf))]
            x_next = self.Metropolis_Hastings(x_0, self.X_inf, it) # [np.array(n.node) for n in self.V]
            x_next = Node(x_next)
            if np.array_equal(np.array(x_next), np.array(x_0)):
                continue
            if self.in_informed(x_next, c_best):
                break
            else:
                continue

        # print("x_next", x_next.node)
        #x_rand = NodeKino(x_next)
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
    def manual_log_like_normal(self, x, data):
        # data = the observation
        # return np.sum(-np.log(x[1] * np.sqrt(2* np.pi) )-((data-x[0])**2) / (2*x[1]**2))
        mean_data = np.mean(data, axis=0) # mean on the columns
        
        cov = [[80., 0.], [0., 80.]] #base value
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

        r = max((20 * math.sqrt((math.log(it) / it))), 1)
        # print("r",r)        
        x_new = self.transition_model(x, r)
        # print("x_new",x_new)
        x_lik = self.manual_log_like_normal(x, data)
        x_new_lik = self.manual_log_like_normal(x_new, data)

        if self.acceptance(
            x_lik + np.log(self.prior(x)), x_new_lik + np.log(self.prior(x_new))
        ):
            return x_new

        return x

    
    @staticmethod
    def SampleUnitBall():
        while True:
            x, y = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y]])
