from irrt_star import *
from rrt_star import *
import argparse



def create_env(env, rnd, n_obs):
  if rnd:
    while len(env.obs_rectangle)!=n_obs:
      rnd_x = random.uniform(env.x_range[0], env.x_range[1])
      rnd_y = random.uniform(env.y_range[0], env.y_range[1])
      rnd_w = random.uniform(0.5, 1.5)
      rnd_h = random.uniform(0.5, 1.5)
    
      env.add_rectangle(rnd_x,rnd_y,rnd_w,rnd_h)
      #print("len",len(env.obs_rectangle))
      #print(rnd_x,rnd_y,rnd_w,rnd_h)

    print("Environment done!")

  else: #fixed environment
    obs_rectangle = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2]
        ]

    for rect in obs_rectangle:
      env.add_rectangle(rect[0], rect[1], rect[2], rect[3])


def main():
    parser = argparse.ArgumentParser(description='Run tests.')
    parser.add_argument('-i','--informed', action='store_true', default=False)
    parser.add_argument('-r','--random', action='store_true', default=False)
    parser.add_argument('-o','--obs', type=int, default=100)
    parser.add_argument('-it','--iter', type=int, default=500)
    parser.add_argument('-c', '--c_best', type=float, default=0.0)

    args = parser.parse_args()
    informed = args.informed
    random = args.random
    obs = args.obs
    iterations = args.iter
    c_best = args.c_best

    x_start = (18, 8)  # Starting node
    x_goal = (37, 18)  # Goal node
    env = Env(x_start=x_start, x_goal=x_goal, delta=0.5)

    # CREATION ENVIRONMENT
    create_env(env, rnd=random, n_obs=obs)

    if informed:
      rrt_star = Informed_RRT_Star(env = env, 
                                  x_start = x_start, 
                                  x_goal = x_goal, 
                                  step_len = 10, 
                                  goal_sample_rate = 0.10, 
                                  search_radius = 12, 
                                  iter_max = iterations,
                                  r_RRT = 10,
                                  stop_at = c_best,
                                  r_goal = 1)
    else:
      rrt_star = RRT_Star(env = env, 
                                x_start = x_start, 
                                x_goal = x_goal, 
                                step_len = 1, 
                                goal_sample_rate = 0.10, 
                                search_radius = 12, 
                                iter_max = iterations,
                                r_RRT = 20,
                                stop_at = c_best)

    rrt_star.planning()

if __name__ == '__main__':
    main()
