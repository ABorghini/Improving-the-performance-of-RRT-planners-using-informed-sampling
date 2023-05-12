from irrt_star import *
from rrt_star import *
from rrt_kino import *
from informed_rrt_kino import *
import argparse
from tqdm import tqdm

def create_env(env, rnd, n_obs, custom_env=False, quad = False):
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
  elif quad:
      obs_rectangle = [
      [40,40,20,20]
    ]
  
  elif rnd:
    rnd_w = 0.5
    rnd_h = 0.5
    while len(obs_rectangle)!=n_obs:
      rnd_x = random.uniform(env.x_range[0]+rnd_w, env.x_range[1]-rnd_w)
      rnd_y = random.uniform(env.y_range[0]+rnd_h, env.y_range[1]-rnd_h)
      
    
      obs_rectangle.append([rnd_x,rnd_y,rnd_w,rnd_h])
      #print("len",len(env.obs_rectangle))
      #print(rnd_x,rnd_y,rnd_w,rnd_h)

    print("Environment done!")

  else: #fixed environment
      obs_rectangle = [
                [60,0,10,20],
                [60,30,10,70],
                [30,0,10,70],
                [30,80,10,20]
            ]
  obs_area = 0
  for rect in obs_rectangle:
    obs_area += rect[2]*rect[3]
    env.add_rectangle(rect[0], rect[1], rect[2], rect[3])
  # print(obs_area)
  return obs_area


def main():
    parser = argparse.ArgumentParser(description='Run tests.')
    parser.add_argument('-i','--informed', action='store_true', default=False)
    parser.add_argument('-r','--random', action='store_true', default=False)
    parser.add_argument('-o','--obs', type=int, default=100)
    parser.add_argument('-it','--iter', type=int, default=50)
    parser.add_argument('-c', '--c_best', type=float, default=0.0)
    parser.add_argument('-k', '--kino', action='store_true', default=False)
    parser.add_argument('-e', '--custom_env', action='store_true', default=False)
    parser.add_argument('-s', '--seed', type=int, default=666)
    parser.add_argument('-es', '--env_seed', type=int, default=666)
    parser.add_argument('-q','--quad', action='store_true', default=False)
    parser.add_argument('-fnr','--fixed_near_radius', action='store_true', default=False)
    parser.add_argument('-mh','--metropolishastings', action='store_true', default=False)

    

    args = parser.parse_args()
    informed = args.informed
    random_ = args.random
    obs = args.obs
    iterations = args.iter
    c_best = args.c_best
    kino = args.kino
    custom_env = args.custom_env
    quad = args.quad
    fnr = args.fixed_near_radius
    mh = args.metropolishastings
    ENV_SEED = args.env_seed
    SEED = args.seed
    random.seed(ENV_SEED)
    np.random.seed(SEED)
    
    # CREATION ENVIRONMENT
    if kino:
      if custom_env:
        x_start = [2,95,0,0]
        x_goal = [80,17,0,0]
        w = 100
        h = 100
        t=0.1
        random_ = False
        delta = 0.6
        
      elif quad:
        random_ = False
        x_start = [35, 50, 0, 0]  # Starting node #50 50 #10 10 
        x_goal = [65, 50, 0, 0] # Goal node #85 85 #17 17
        #obs_dim = 0.5 #2.5 #0.5
        w=100
        h=100
        t=0.1
        delta = 0.2
        # path = f'real_simulations/quadratino'
      elif random_:
        x_start = [10,10,0,0]
        x_goal = [17,17,0,0] 
        w = 20
        h = 20
        t=0.02
        delta = 0.2
      else:
        x_start = [2,2,0,0]
        x_goal = [98,98,0,0] #[50,65,0,0] #  
        w = 100
        h = 100
        t=0.1
        delta = 0.6

      env = EnvKino(x_start=x_start, x_goal=x_goal, delta=delta, w=w, h=h, thickness=t)
      obs_area = create_env(env, rnd=random_, n_obs=obs, custom_env=custom_env, quad=quad)

      if informed:
          # SEED = 800
        # seeds = [s for s in range(0,1000,100)]
        # for SEED in seeds:
        #   random.seed(SEED)
        #   np.random.seed(SEED)
        rrt_star = Informed_RRT_Star_Kino(env = env, 
                                    x_start = x_start, 
                                    x_goal = x_goal,
                                    iter_max = iterations,
                                    state_dims = 4,
                                    input_dims = 2,
                                    state_limits = [[0, 100], [0, 100], [-10, 10], [-10, 10]],
                                    input_limits = [[-5, 5], [-5, 5]],
                                    stop_at= c_best,
                                    custom_env=custom_env,
                                    seed=SEED)
          # rrt_star.planning()
      else:
        # seeds = [s for s in range(0,1000,100)]
        # for SEED in seeds:
        #   random.seed(SEED)
        #   np.random.seed(SEED)
        rrt_star = RRT_Star_Kino(env = env, 
                                  x_start = x_start, 
                                  x_goal = x_goal, 
                                  iter_max = iterations,
                                  state_dims = 4,
                                  input_dims = 2,
                                  state_limits = [[0, 100], [0, 100], [-10, 10], [-10, 10]],
                                  input_limits = [[-5, 5], [-5, 5]],
                                  stop_at= c_best,
                                  custom_env=custom_env,
                                  seed=SEED)
          # rrt_star.planning()
    else: #not kino
      if custom_env:
        x_start = [2,95]
        x_goal = [80,17]   # Goal node
        w = 100
        h = 100
        delta = 0.2
        t=0.1
        step_len = 10
        random_ = False
        r_RRT = 25     
        #deve essere maggiore di 122.30
      elif quad:
        random_ = False
        x_start = [35, 50]  # Starting node #50 50 #10 10 
        x_goal = [65, 50] # Goal node #85 85 #17 17
        #obs_dim = 0.5 #2.5 #0.5
        w=100
        h=100
        t=0.1
        delta = 0.2
        r_RRT = 20
        step_len = 3
        random_ = False
      elif random_:
        x_start = [10,10]
        x_goal = [17,17] 
        w = 20
        h = 20
        t=0.02
        r_RRT = 5
        step_len = 1
        delta = 0.1
      else:
        x_start = [2,2]
        x_goal = [98,98]
        w = 100
        h = 100
        step_len=3
        r_RRT = 25
        delta = 0.2
        t=0.1

      env = Env(x_start=x_start, x_goal=x_goal, delta=delta, w=w, h=h, thickness=t)
      obs_area = create_env(env, rnd=random_, n_obs=obs, custom_env=custom_env, quad=quad)

      d = len(x_start)
      if not fnr:
        r_RRT = 2*np.power(1+1/d,1/d)*np.power((((w*h)-obs_area)/np.pi),1/d)
        print("computed neighbour radius: ", r_RRT)
  
      if informed:
        seeds = [s for s in range(500,1000,100)]
        # for SEED in seeds:
        #   # SEED = 666
        #   random.seed(ENV_SEED)
        #   np.random.seed(SEED)
        rrt_star = Informed_RRT_Star(env = env, 
                        x_start = x_start, 
                        x_goal = x_goal, 
                        step_len = step_len, #5 
                        goal_sample_rate = 0.10, 
                        search_radius = 12, 
                        iter_max = iterations,
                        r_RRT = r_RRT,
                        fixed_near_radius=fnr,
                        r_goal = 1,
                        stop_at = c_best,
                        rnd=random_,
                        n_obs=obs,
                        custom_env=custom_env,
                        seed=SEED,
                        env_seed=ENV_SEED,
                        mh=mh)
          # rrt_star.planning()
      else:
        # seeds = [s for s in range(0,1000,100)]
        # for SEED in seeds:
        #   random.seed(ENV_SEED)
        #   np.random.seed(SEED)
        rrt_star = RRT_Star(env = env, 
                                  x_start = x_start, 
                                  x_goal = x_goal, 
                                  step_len = step_len, 
                                  goal_sample_rate = 0.10, 
                                  search_radius = 12, 
                                  iter_max = iterations,
                                  r_RRT = r_RRT,
                                  fixed_near_radius=fnr,
                                  r_goal = 1,
                                  stop_at = c_best,
                                  rnd=random_,
                                  n_obs=obs,
                                  custom_env=custom_env,
                                  seed=SEED,
                                  env_seed=ENV_SEED)
      # rrt_star.planning()  
    rrt_star.planning()  

if __name__ == '__main__':
    main()