from irrt_star import *
from rrt_star import *
from rrt_kino import *
from informed_rrt_kino import *
import argparse

def create_env(env, rnd, n_obs, delta=0.6, kino=False, custom_env=False):
  obs_rectangle = []
  if custom_env:
    obs_rectangle = [
        [0,85,5,5],
        [18,93,15,7],
        [60,93,25,7],
        [0, 50,5,20],
        [0,10,15,8],
        [15, 10, 5, 25],
        [15, 20, 25, 5],
        [70, 8, 5, 17],
        [55, 8, 4, 60],
        [35, 40, 50, 4],
        [70, 25, 17, 5],
        [85, 20, 5, 10],
        [70, 8, 10, 5],
        [80, 60, 20, 5],
        [75, 55, 7, 20],
        [20, 55, 17, 15],
        [26, 70, 4, 10],
        [21, 80, 37 ,4]
      ]
  
  elif rnd:
    rnd_w = 0.5
    rnd_h = 0.5
    while len(obs_rectangle)!=n_obs:
      rnd_x = random.uniform(env.x_range[0], env.x_range[1]-delta)
      rnd_y = random.uniform(env.y_range[0], env.y_range[1]-delta)
      
    
      obs_rectangle.append([rnd_x,rnd_y,rnd_w,rnd_h])
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
    parser.add_argument('-it','--iter', type=int, default=500)
    parser.add_argument('-c', '--c_best', type=float, default=0.0)
    parser.add_argument('-k', '--kino', action='store_true', default=False)
    parser.add_argument('-e', '--custom_env', action='store_true', default=False)
    parser.add_argument('-s', '--seed', type=int, default=666)
    parser.add_argument('-es', '--env_seed', type=int, default=666)

    args = parser.parse_args()
    informed = args.informed
    random_ = args.random
    obs = args.obs
    iterations = args.iter
    c_best = args.c_best
    kino = args.kino
    custom_env = args.custom_env
    SEED = args.seed
    ENV_SEED = args.env_seed
    np.random.seed(SEED)
    random.seed(ENV_SEED)

    # CREATION ENVIRONMENT
    if kino:
      if custom_env:
        x_start = [2,95,0,0]
        x_goal = [80,17,0,0]
        w = 100
        h = 100
      else:
        x_start = [2,2,0,0]
        x_goal = [98,98,0,0] #[50,65,0,0] #  
        w = 100
        h = 100

      env = EnvKino(x_start=x_start, x_goal=x_goal, delta=0.5, w=w, h=h)
      create_env(env, rnd=random_, n_obs=obs, kino=kino, custom_env=custom_env)

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
                                    stop_at= c_best,
                                    custom_env=custom_env)
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
                                  stop_at= c_best,
                                  custom_env=custom_env)
        
    else: #not kino
      if custom_env:
        random_ = False
        x_start = [2,95]
        x_goal = [80,17]   # Goal node
        w = 100
        h = 100
        step_length = 10
      else:
        x_start = [10,10]
        x_goal = [18,18]
        w = 20
        h = 20
        step_length = 1

      env = Env(x_start=x_start, x_goal=x_goal, delta=0.6, w=w,h=h)

      create_env(env, rnd=random_, n_obs=obs, custom_env=custom_env)

      if informed:
        rrt_star = Informed_RRT_Star(env = env, 
                                    x_start = x_start, 
                                    x_goal = x_goal, 
                                    step_len = step_length, 
                                    goal_sample_rate = 0.10, 
                                    search_radius = 12, 
                                    iter_max = iterations,
                                    r_RRT = 10,
                                    r_goal = 1,
                                    stop_at = c_best,
                                    rnd=random_,
                                    n_obs = obs,
                                    custom_env=custom_env)
      else:
        rrt_star = RRT_Star(env = env, 
                                  x_start = x_start, 
                                  x_goal = x_goal, 
                                  step_len = step_length, 
                                  goal_sample_rate = 0.10, 
                                  search_radius = 12, 
                                  iter_max = iterations,
                                  r_RRT = 10,
                                  r_goal = 1,
                                  stop_at = c_best,
                                  rnd=random_,
                                  n_obs = obs,
                                  custom_env=custom_env)
    rrt_star.planning()

if __name__ == '__main__':
    main()