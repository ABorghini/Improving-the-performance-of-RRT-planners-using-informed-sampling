import numpy as np
import math
from sympy import *

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
    
    def equals(self, __o: object) -> bool:
        if self.x == __o.x and self.y == __o.y:
            return True
        return False

class Env:
    def __init__(self, x_start, x_goal, w = 50, h = 30, thickness = 1, delta = 0.5):
        self.x_range = (0, w)
        self.y_range = (0, h)
        self.x_start = x_start
        self.x_goal = x_goal
        self.delta = delta
        self.thickness = thickness
        self.obs_boundary = self.boundaries()
        self.obs_rectangle = []


    def boundaries(self):
        obs_boundary = [
            [self.x_range[0], self.y_range[0], self.thickness, self.y_range[1]],
            [self.x_range[0], self.y_range[1], self.x_range[1], self.thickness],
            [self.thickness, self.x_range[0], self.x_range[1], self.thickness],
            [self.x_range[1], self.thickness, self.thickness, self.y_range[1]]
        ]
        return obs_boundary

    def add_rectangle(self, x, y, w, h):
        if (0 <= self.x_start[0] - (x - self.delta) <= w + 2 * self.delta \
            and 0 <= self.x_start[1] - (y - self.delta) <= h + 2 * self.delta) \
                or (0 <= self.x_goal[0] - (x - self.delta) <= w + 2 * self.delta \
                    and 0 <= self.x_goal[1] - (y - self.delta) <= h + 2 * self.delta) :
                        return
        
        else:
            self.obs_rectangle.append([x,y,w,h])


    def get_obs_vertex(self):
        delta = self.delta
        obs_list = []

        for (ox, oy, w, h) in self.obs_rectangle:
            vertex_list = [[ox - delta, oy - delta],
                           [ox + w + delta, oy - delta],
                           [ox + w + delta, oy + h + delta],
                           [ox - delta, oy + h + delta]]
            obs_list.append(vertex_list)

        return obs_list

    def is_intersect_rec(self, start, end, o, d, a, b):
        v1 = [o[0] - a[0], o[1] - a[1]]
        v2 = [b[0] - a[0], b[1] - a[1]]
        v3 = [-d[1], d[0]]

        div = np.dot(v2, v3)

        if div == 0:
            return False

        t1 = np.linalg.norm(np.cross(v2, v1)) / div
        t2 = np.dot(v1, v3) / div

        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1]))
            dist_obs = self.get_dist(start, shot)
            dist_seg = self.get_dist(start, end)
            if dist_obs <= dist_seg:
                return True

        return False


    def isCollision(self, start, end):
        if self.is_inside_obs(start) or self.is_inside_obs(end):
            return True

        o, d = self.get_ray(start, end)
        obs_vertex = self.get_obs_vertex()

        for (v1, v2, v3, v4) in obs_vertex:
            if self.is_intersect_rec(start, end, o, d, v1, v2):
                return True
            if self.is_intersect_rec(start, end, o, d, v2, v3):
                return True
            if self.is_intersect_rec(start, end, o, d, v3, v4):
                return True
            if self.is_intersect_rec(start, end, o, d, v4, v1):
                return True

        return False

    def is_inside_obs(self, node):
        delta = 0.5

        for (x, y, w, h) in self.obs_rectangle:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        for (x, y, w, h) in self.obs_boundary:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True

        return False


    @staticmethod
    def get_ray(start, end):
        orig = [start.x, start.y]
        direc = [end.x - start.x, end.y - start.y]
        return orig, direc

    @staticmethod
    def get_dist(start, end):
        return math.hypot(end.x - start.x, end.y - start.y)




