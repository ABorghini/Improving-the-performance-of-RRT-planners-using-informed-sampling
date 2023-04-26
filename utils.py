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
import matplotlib.animation as animation
from PIL import Image

#infinity symbol on latex: \infty

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def animate(algo):
    img_list= []
    ims = []
    imgs = os.listdir(algo.plotting_path)
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    for i in imgs:
        img = Image.open(f'{algo.plotting_path}/{i}')
        img_list.append(img)
    ims = [[plt.imshow(i, animated=True)] for i in img_list] #np.transpose(i,(1,2,0))
    ani = animation.ArtistAnimation(fig, ims, blit=True)

    ani.save(f'{algo.plotting_path}/{algo.name}_{algo.iter_max}.gif')

def plot(algo, iteration, x_center=None, c_best=np.inf, dist=None, theta=None):

    plt.cla()

    if c_best==np.inf:
        c_print = r'$\infty$'
    else:
        c_print = "{:.2f}".format(c_best)

    t_print = "{:.2f}".format(algo.duration)

    if algo.stop_at == 0:
        plot_grid(f"{algo.name}",f"it = {str(iteration)}/{str(algo.iter_max)}, nodes = {str(len(algo.V))}, c_best = {c_print}", algo)
    else:
        plot_grid(f"{algo.name}",f"it = {str(iteration)}, goal_cost = {str(algo.stop_at)}, nodes = {str(len(algo.V))}, c_best = {c_print}", algo)


    lines = [[(n.x, n.y),(n.parent.x, n.parent.y)] for n in algo.V if n.parent]

    lc = mc.LineCollection(lines, linewidths=1, colors=(23/255, 106/255, 255/255, 1))
    algo.ax.add_collection(lc)

#    if algo.stop_at == 0:
    dimension = len(str(algo.iter_max))
    zero_to_add = dimension - len(str(iteration))
    added_zeros = '0'*zero_to_add
    nx, ny = zip(*[[n.x,n.y] for n in algo.V])
    plt.plot(nx,ny, marker='o', color= 'c', linestyle='', markersize=2)
    if algo.sol > 0:
        nx, ny = zip(*[[n.x,n.y] for n in algo.path])
        plt.plot(nx,ny, marker='o', color= 'r', markersize=2)
    
    if algo.name == 'IRRT_star' or algo.name == 'IRRTK_star':
        if c_best != np.inf and not algo.mh:
            draw_ellipse(x_center, c_best, dist, theta)
    # else:
    #     dimension = 5 #default set to N = X0'000
    #     zero_to_add = dimension - len(str(iteration))
    #     added_zeros = '0'*zero_to_add

    #check if the directory exists, else create it
    create_dir(algo.plotting_path)

    plt.savefig(f'{algo.plotting_path}/img_{added_zeros}{iteration}')

    plt.pause(0.01)


def plot_grid(name, info, rrt):

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

    if rrt.name == "RRTK_star" or rrt.name == "IRRTK_star":
        plt.plot(rrt.x_start.node[0], rrt.x_start.node[1], "bs", linewidth=3)
        plt.plot(rrt.x_goal.node[0], rrt.x_goal.node[1], "rs", linewidth=3)
    else:
        plt.plot(rrt.x_start.x, rrt.x_start.y, "bs", linewidth=3)
        plt.plot(rrt.x_goal.x, rrt.x_goal.y, "rs", linewidth=3)

    #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
    plt.title(info)
    #plt.xlabel(info)
    #plt.axis("equal")
    fig = plt.gcf()
    fig.suptitle(name, fontsize=13, fontweight='bold')

    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    start, end = ax.get_ylim()
    start*=0
    end-=1
    #print(start,end)
    if rrt.name == "RRTK_star" or rrt.name == "IRRTK_star":
        stepsize = 20
    else:
        if rrt.rnd:
            stepsize = 5
        else:
            stepsize = 20
    ax.yaxis.set_ticks(np.arange(start, end, stepsize))
    ax.xaxis.set_ticks(np.arange(start, end, stepsize))

    #ax.set_visible('False')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    #ax.get_xaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([])
    #plt.axis("off")

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


def plot_kino(rrtk, iteration, x_center=None, c_best=np.inf, tau_star=np.inf, dist=None, theta=None):
    step = 1.0
    path = rrtk.path
    path.reverse()

    plt.cla()

    if c_best==np.inf:
        c_print = r'$\infty$'
    else:
        c_print = "{:.2f}".format(c_best)

    t_print = "{:.2f}".format(tau_star)

    print("plot_grid")
    if rrtk.stop_at > 0:
        plot_grid(f"{rrtk.name}",f"it = {str(iteration)}, nodes = {str(len(rrtk.V))}, c_best = {c_print}, tau_star = {t_print}", rrtk)    
    else:
        plot_grid(f"{rrtk.name}",f"it = {str(iteration)}/{str(rrtk.iter_max)}, nodes = {str(len(rrtk.V))}, c_best = {c_print}, tau_star = {t_print}", rrtk)
    states_list = []
    for idx, node in enumerate(path):
        print(idx)
        if idx == len(path)-1:
            break
        t_goal = rrtk.eval_arrival_time(node, path[idx+1])
        times = np.arange(0, t_goal/step)*step
        states = lambdify([rrtk.x, rrtk.t], rrtk.states(t_goal, rrtk.t, rrtk.x, node, path[idx+1]), "numpy")
        
        states_list.extend([states(rrtk.t, t)[0:2] for t in times])

    # generated nodes
    node_listx, node_listy, vx, vy = zip(*[np.array(node.node,dtype=np.float64) for node in rrtk.V])
    # print(*[list(node_listx),list(node_listy)])
    plt.plot(node_listx, node_listy, marker='o', color= 'plum', linestyle='', markersize=2)
    # plt.quiver(*[list(node_listx),list(node_listy)],list(vx),list(vy),color='r')
    # generated nodes after first solution
    if len(rrtk.V) > rrtk.firstsol:
        node_listx, node_listy, vx, vy = zip(*[np.array(node.node,dtype=np.float64) for node in rrtk.V[rrtk.firstsol:]])
        plt.plot(node_listx, node_listy, marker='o', color= 'lightskyblue', linestyle='', markersize=2)
        # plt.quiver(*[list(node_listx),list(node_listy)],list(vx),list(vy),color='r')

    x, y = zip(*states_list)
    if len(path) > 2:
        x_, y_, vx_, vy_ = zip(*path[1:-1])
    else:
        x_ = []
        y_ = []


    plt.plot(x,y, color='c')

    plt.plot(x_,y_, color='g', marker='o', linestyle='', markersize=4)
    # plt.quiver(*[list(x_),list(y_)],list(vx_),list(vy_),color='r',scale=5)
    # plt.show()
    # if rrtk.name == "IRRTK_star":
    #     node_listx, node_listy, _, _ = zip(*[node.node for node in rrtk.X_inf])
    #     plt.plot(node_listx, node_listy, marker='o', color= 'red', linestyle='', markersize=2)

    create_dir(rrtk.plotting_path)
    if iteration == rrtk.iter_max:
        plt.savefig(f'{rrtk.plotting_path}/img_{rrtk.sol + 1}')
    else: 
        plt.savefig(f'{rrtk.plotting_path}/img_{rrtk.sol}')
    plt.pause(0.01)
    return
