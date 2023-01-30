from turtle import color
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import collections as mc
import os
import sys
import numpy as np
import math
from scipy.spatial.transform import Rotation as Rot


def plot(algo, iteration, x_center=None, c_best=np.inf, dist=None, theta=None):

    # print("V",len(rrt.V))
    plt.cla()
    # if iteration == 0:
    #     rrt.in_plot = []
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
