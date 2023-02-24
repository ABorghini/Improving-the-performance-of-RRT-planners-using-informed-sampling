from turtle import color
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import collections as mc
import os
import sys
import numpy as np
import math
from scipy.spatial.transform import Rotation as Rot
from sympy.abc import x
from sympy import *


def plot(algo, iteration, x_center=None, c_best=np.inf, dist=None, theta=None):

    plt.cla()

    plot_grid(f"{algo.name}, N = {str(algo.iter_max)}", algo)

    lines = [[(n.x, n.y),(n.parent.x, n.parent.y)] for n in algo.V if n.parent]

    lc = mc.LineCollection(lines, linewidths=1, colors=(23/255, 106/255, 255/255, 1))
    algo.ax.add_collection(lc)

    if c_best != np.inf:
        draw_ellipse(x_center, c_best, dist, theta)

    plt.pause(0.01)


def plot_grid(name, rrt):

    for (ox, oy, w, h) in rrt.obs_boundary:
        rrt.ax.add_patch(
            patches.Rectangle(
                (ox, oy), w, h,
                edgecolor='black',
                facecolor='black',
                fill=True
            )
        )

    for (ox, oy, w, h) in rrt.obs_rectangle:
        rrt.ax.add_patch(
            patches.Rectangle(
                (ox, oy), w, h,
                edgecolor='black',
                facecolor='black',
                fill=True
            )
        )

    if rrt.name == "RRTK*":
        plt.plot(rrt.x_start.node[0], rrt.x_start.node[1], "bs", linewidth=3)
        plt.plot(rrt.x_goal.node[0], rrt.x_goal.node[1], "rs", linewidth=3)
    else:
        plt.plot(rrt.x_start.x, rrt.x_start.y, "bs", linewidth=3)
        plt.plot(rrt.x_goal.x, rrt.x_goal.y, "rs", linewidth=3)

    plt.title(name)
    plt.axis("equal")

def draw_ellipse(x_center, c_best, dist, theta):
    a = math.sqrt(c_best**2 - dist**2) / 2.0
    b = c_best / 2.0
    angle = math.pi / 2.0 - theta
    cx = x_center[0]
    cy = x_center[1]
    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
    fx = rot @ np.array([x, y])
    px = np.array(fx[0, :] + cx).flatten()
    py = np.array(fx[1, :] + cy).flatten()
    plt.plot(cx, cy, ".b")
    plt.plot(px, py, linestyle='--', color='darkorange', linewidth=2)


def plot_kino(rrtk, x_center=None, c_best=np.inf, dist=None, theta=None):
    step = 1.0
    path = rrtk.path
    path.reverse()

    plt.cla()

    print("plot_grid")
    plot_grid(f"{rrtk.name}, N = {str(rrtk.iter_max)}", rrtk)
    states_list = []
    for idx, node in enumerate(path):
        print(idx)
        if idx == len(path)-1:
            break
        t_goal = rrtk.eval_arrival_time(node, path[idx+1])
        times = np.arange(0, t_goal/step)*step
        states = lambdify([rrtk.x, rrtk.t], rrtk.states(t_goal, rrtk.t, rrtk.x, node, path[idx+1]), "numpy")
        
        states_list.extend([states(rrtk.t, t)[0:2] for t in times])

    print(states_list)
    x, y = zip(*states_list)
    if len(path) > 2:
        x_, y_, _, _ = zip(*path[1:-1])
    else:
        x_ = []
        y_ = []
    plt.plot(x,y, color='c')
    plt.plot(x_,y_, color='g', marker='o', linestyle='', markersize=4)
    node_listx, node_listy, _, _ = zip(*[node.node for node in rrtk.V])
    # print(rrtk.V)
    plt.plot(node_listx, node_listy, marker='o', color= 'r', linestyle='', markersize=4)
    plt.show()
    plt.pause(0.01)
    return
