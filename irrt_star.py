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


class Informed_RRT_Star(RRT_Star):
    def __init__(self, env, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max, r_RRT, r_goal, stop_at, rnd):
        
        super().__init__(env, x_start, x_goal, step_len,
                        goal_sample_rate, search_radius,
                        iter_max, r_RRT, r_goal, stop_at,rnd)   
        self.name = 'IRRT_star'
        self.X_soln = set()

        if rnd:
            self.plotting_path = f'{self.name}_imgs_max_{self.iter_max}_randEnv'
        else:
            self.plotting_path = f'{self.name}_imgs_max_{self.iter_max}_fixedEnv'


    def planning(self):
        theta, dist, x_center, C, x_best = self.init()
        c_best = np.inf
   
        x_best = self.x_start
        i = 0
        while i<self.iter_max and c_best >= self.stop_at:
            if self.X_soln:
                cost = {node: self.Cost(node, self.x_goal) for node in self.X_soln}
                x_best = min(cost, key=cost.get)
                c_best = cost[x_best]
                print("c_best", c_best)
                if c_best == self.stop_at:
                    break

            x_rand = self.Sample(c_best, dist, x_center, C)
            x_nearest = self.Nearest(x_rand)
            x_new = self.Steer(x_nearest, x_rand)

            if not self.env.isCollision(x_nearest, x_new):
                X_near = self.Near(self.V, x_new, self.r_RRT) # r_RRT
                c_min = self.Cost(x_nearest, x_new)

                # choose parent
                x_new, _ = self.ChooseParent(X_near, x_new, c_min)

                self.V.append(x_new)

                # Rewire
                self.Rewire(X_near, x_new)
                
                x_new = self.V[self.V.index(x_new)]
               
                if self.InGoalRegion(x_new):
                    if not self.env.isCollision(x_new, self.x_goal):
                        self.X_soln.add(x_new)

            # print("iter",i)
            if i % 20 == 0 or i == self.iter_max-1:
                #print("iter", i)
                plot(self, i, self.iter_max, self.plotting_path, x_center=x_center, c_best=c_best, dist=dist, theta=theta)
            
            i+=1

        self.path = self.ExtractPath(x_best)
        plot(self, i, self.iter_max, self.plotting_path, x_center=x_center, c_best=c_best, dist=dist, theta=theta)
        plt.plot([x for x, _ in self.path], [y for _, y in self.path], color=(1.0,0.0,0.0,1))
        plt.savefig(f'{self.plotting_path}/img_{i}1')
        plt.pause(0.001)
        plt.show()
        animate(self, self.iter_max, self.plotting_path)


    def Near(self, V, x_new, search_radius = 20):
        n = len(V) + 1
        # r2 = (search_radius * math.sqrt((math.log(n) / n))) ** 2
        r = min((search_radius * math.sqrt((math.log(n) / n))), self.step_len)
        # DA RIVEDERE STA COSA DEL RAGGIO PD
        #print("r2",r2)
        self.step_len = r
        r2 = r**2
        dist_table = [(n.x - x_new.x) ** 2 + (n.y - x_new.y) ** 2 for n in V]
        X_near = [v for v in V if dist_table[V.index(v)] <= r2 and not self.env.isCollision(v, x_new)]
        
        return X_near

    def Sample(self, c_max, c_min, x_center, C):

        if c_max < np.inf: #at least a solution has been found
            
            # print(c_min, c_max)

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
    
    @staticmethod
    def SampleUnitBall():
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y]])
