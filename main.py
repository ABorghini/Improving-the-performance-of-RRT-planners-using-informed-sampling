from irrt_star import *
from rrt_star import *
from rrt_kino import *
from informed_rrt_kino import *
import argparse

SEED = 777

random.seed(SEED)
np.random.seed(SEED)

def create_env(env, rnd, n_obs, kino=False):
  if rnd:
    while len(env.obs_rectangle)!=n_obs:
      rnd_x = random.uniform(env.x_range[0], env.x_range[1]-0.5)
      rnd_y = random.uniform(env.y_range[0], env.y_range[1]-0.5)
      rnd_w = random.uniform(0.5, 0.5)
      rnd_h = random.uniform(0.5, 0.5)
    
      env.add_rectangle(rnd_x,rnd_y,rnd_w,rnd_h)
      #print("len",len(env.obs_rectangle))
      #print(rnd_x,rnd_y,rnd_w,rnd_h)

    print("Environment done!")

  else: #fixed environment
    if kino:
      obs_rectangle = [
                [60,0,10,20],
                [60,30,10,70],
                [30,0,10,70],
                [30,80,10,20]
            ]
    else:
      obs_rectangle = [
              [12, 0, 2, 4],
              [12, 6, 2, 14],
              [6, 0, 2, 14],
              [6, 16, 2, 4]
          ]
      # [
      #         [6, 8, 3, 1],
      #         [7, 15, 3, 2],
      #         [10, 5, 1, 8],
      #         [13, 9, 4, 1]
      #     ]
      # [
      #         [14, 12, 8, 2],
      #         [18, 22, 8, 3],
      #         [26, 7, 2, 12],
      #         [32, 14, 10, 2]
      #     ]

    for rect in obs_rectangle:
      env.add_rectangle(rect[0], rect[1], rect[2], rect[3])


def main():
    parser = argparse.ArgumentParser(description='Run tests.')
    parser.add_argument('-i','--informed', action='store_true', default=False)
    parser.add_argument('-r','--random', action='store_true', default=False)
    parser.add_argument('-o','--obs', type=int, default=100)
    parser.add_argument('-it','--iter', type=int, default=50)
    parser.add_argument('-c', '--c_best', type=float, default=0.0)
    parser.add_argument('-k', '--kino', action='store_true', default=False)

    args = parser.parse_args()
    informed = args.informed
    random_ = args.random
    obs = args.obs
    iterations = args.iter
    c_best = args.c_best
    kino = args.kino

    # CREATION ENVIRONMENT
    if kino:
      SEED = 666
      random.seed(SEED)
      np.random.seed(SEED)

      x_start = [2,2,0,0]
      x_goal = [50,65,0,0] # [98,98,0,0]
      env = EnvKino(x_start=x_start, x_goal=x_goal, delta=0.5)
      create_env(env, rnd=random_, n_obs=obs, kino=kino)

      if informed:
        rrt_star = Informed_RRT_Star_Kino(env = env, 
                                    x_start = x_start, 
                                    x_goal = x_goal,
                                    max_radius = 100, 
                                    iter_max = iterations,
                                    state_dims = 4,
                                    input_dims = 2,
                                    state_limits = [[0, 100], [0, 100], [-10, 10], [-10, 10]],
                                    input_limits = [[-5, 5], [-5, 5]],
                                    stop_at= c_best)
      else:
        rrt_star = RRT_Star_Kino(env = env, 
                                  x_start = x_start, 
                                  x_goal = x_goal, 
                                  max_radius = 100, 
                                  iter_max = iterations,
                                  state_dims = 4,
                                  input_dims = 2,
                                  state_limits = [[0, 100], [0, 100], [-10, 10], [-10, 10]],
                                  input_limits = [[-5, 5], [-5, 5]],
                                  stop_at= c_best)
        
    else: #not kino

      x_start = (2, 2)  # Starting node
      x_goal = (18, 18)  # Goal node
      env = Env(x_start=x_start, x_goal=x_goal, delta=0.5)

      create_env(env, rnd=random_, n_obs=obs)

      if informed:
        rrt_star = Informed_RRT_Star(env = env, 
                                    x_start = x_start, 
                                    x_goal = x_goal, 
                                    step_len = 1, 
                                    goal_sample_rate = 0.10, 
                                    search_radius = 12, 
                                    iter_max = iterations,
                                    r_RRT = 10,
                                    r_goal = 1,
                                    stop_at = c_best,
                                    rnd=random_)
      else:
        rrt_star = RRT_Star(env = env, 
                                  x_start = x_start, 
                                  x_goal = x_goal, 
                                  step_len = 1, 
                                  goal_sample_rate = 0.10, 
                                  search_radius = 12, 
                                  iter_max = iterations,
                                  r_RRT = 20,
                                  r_goal = 1,
                                  stop_at = c_best,
                                  rnd=random_)
    rrt_star.planning()

if __name__ == '__main__':
    main()