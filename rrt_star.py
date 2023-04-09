import os
import sys
import math
from turtle import width
import numpy as np
import matplotlib.pyplot as plt

from env import Node, Env
from utils import plot, animate
import time as timing


class RRT_Star:
    def __init__(self, env, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max, r_RRT, r_goal, stop_at, rnd, n_obs):

        self.name = 'RRT_star'
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.r_RRT = r_RRT
        self.r_goal = r_goal
        self.env = env
        self.stop_at = stop_at

        if self.stop_at!=0:
            self.iter_max = 10000

        self.fig, self.ax = plt.subplots()
        self.delta = self.env.delta
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.V = [self.x_start]
        # self.X_soln = set()
        self.path = None
        rnd_path = 'randEnv' if rnd else 'fixedEnv'
        c_path = f'_C_{self.stop_at}_' if self.stop_at!=0 else ''
        n_path = f'_N_{self.iter_max}_' if self.stop_at==0 else ''
        obs_path = f'_O_{n_obs}' if rnd else ''

        self.plotting_path = f'{self.name}{n_path}{c_path}{rnd_path}{obs_path}'

        self.duration = 0 #to add for the graphic analysis without the plots


    def init(self):
        cMin, theta = self.get_distance_and_angle(self.x_start, self.x_goal)
        C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        xCenter = np.array([[(self.x_start.x + self.x_goal.x) / 2.0],
                            [(self.x_start.y + self.x_goal.y) / 2.0]])
        x_best = self.x_start

        return theta, cMin, xCenter, C, x_best


    def planning(self):
        theta, dist, x_center, C, x_best = self.init()
        c_best = np.inf
        first_enter = 0
        x_best = self.x_start
        i = 0
        #ts = timing.time()
        while i<self.iter_max and c_best > self.stop_at:

            x_rand = self.SampleFromSpace()
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

                x = self.V[self.V.index(x_new)]

                if x.equals(self.x_goal): #forse da eliminare
                    if first_enter==0:
                        print('entrato alla iterazione', i)
                        first_enter+=1
                    x_best = x
                    c_best = self.Cost(x)

            if i % 20 == 0 or i == self.iter_max-1:
                plot(self, i, c_best=c_best)
            i+=1


        x_best, c_best = self.search_best()
        self.path = self.ExtractPath(x_best)
        plot(self, i, c_best=c_best)
        plt.plot([x for x, _ in self.path], [y for _, y in self.path], color=(1.0,0.0,0.0,1.0))
        plt.savefig(f'{self.plotting_path}/img_{i}1')
        plt.pause(0.001)
        plt.show()
        animate(self)

    #initializes a new node in the direction of x_goal, distant at most step_len
    #from x_start
    def Steer(self, x_start, x_goal):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        dist = min(self.step_len, dist)

        node_new = Node((x_start.x + dist * math.cos(theta),
                         x_start.y + dist * math.sin(theta)))
        node_new.parent = x_start

        return node_new

    def Nearest(self, x_rand):
        return self.V[int(np.argmin([(n.x - x_rand.x) ** 2 + (n.y - x_rand.y) ** 2 for n in self.V]))]

    def Near(self, V, x_new, search_radius = 20):
        n = len(V) + 1
        r = min((search_radius * math.sqrt((math.log(n) / n))), self.step_len)
        # self.step_len = r
        r2 = r**2
        dist_table = [(n.x - x_new.x) ** 2 + (n.y - x_new.y) ** 2 for n in V]
        X_near = [v for v in V if dist_table[V.index(v)] <= r2 and not self.env.isCollision(v, x_new)]

        return X_near


    def inside_boundaries(self, x_rand):
        if x_rand is not None:
            return self.x_range[0] + self.delta <= x_rand[0] <= self.x_range[1] - self.delta and \
                    self.y_range[0] + self.delta <= x_rand[1] <= self.y_range[1] - self.delta
        else:
            return False

    def ChooseParent(self, X_near, x_new, c_min):
        for x_near in X_near:
            c_new = self.Cost(x_near, x_new)
            if c_new < c_min and not self.env.isCollision(x_near, x_new):
                x_new.parent = x_near
                c_min = c_new
        return x_new, c_min

    def Rewire(self, X_near, x_new):
        for x_near in X_near:
            c_near = self.Cost(x_near)
            c_new = self.Cost(x_new, x_near)
            if c_new < c_near:
                self.V[self.V.index(x_near)].parent = x_new
                # x_near.parent = x_new

    def search_best(self):
        distances = [(n.x - self.x_goal.x) ** 2 + (n.y - self.x_goal.y) ** 2 for n in self.V]
        r2 = self.step_len**2
        indeces = [i for i in range(len(distances)) if distances[i] <= r2]

        if len(indeces)==0:
            return self.x_goal, np.inf
        cost, idxs = zip(*[[self.Cost(self.V[idx], self.x_goal), idx] for idx in indeces])
        c_i = np.argmin(np.array(cost))
        best_index = idxs[c_i]
        x_best = self.V[best_index]
        c_best = cost[c_i]
        return x_best, c_best



    def SampleFromSpace(self):
        delta = self.delta

        if np.random.random() > self.goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.x_goal


    def ExtractPath(self, node):
        path = [[self.x_goal.x, self.x_goal.y]]

        while node.parent:
            path.append([node.x, node.y])
            node = node.parent

        path.append([self.x_start.x, self.x_start.y])

        return path

    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        # axes of the hyperellipsoid
        a1 = np.array([[(x_goal.x - x_start.x) / L],
                       [(x_goal.y - x_start.y) / L]])

        # first column of the identity matrix
        e1 = np.array([[1.0], [0.0]])

        M = a1 @ e1.T

        U, _, V_T = np.linalg.svd(M, full_matrices=True, compute_uv=True)
        C = U @ np.diag([1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T

        return C


    @staticmethod
    def Line(x_start, x_goal):
        return math.hypot(x_goal.x - x_start.x, x_goal.y - x_start.y)


    def Cost(self, node, node2 = None):
        if node == self.x_start:
            return 0.0

        if node.parent is None:
            return np.inf

        if node2 is None:
            cost = 0.0
        else:
            cost = self.Line(node, node2)
        while node.parent:
            cost += self.Line(node, node.parent)
            node = node.parent

        return cost


    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)
