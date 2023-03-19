# Improving-the-performance-of-RRT-planners-using-informed-sampling

## Update
The informed version of the kinodynamic RRT* has been uploaded.

All the files were updated to a better plotting.

The implementation of the Kinodynamic RRT* has been improved, now it is faster.

## Work in progress
The current implementation of the Kinodynamic RRT* (together with its informed version) do not support the configurations specified below, to run it simply write in the terminal:


## About
In this implementation we present the rrt* algorithm in 4 different ways. The implementations are:
* RRT*
* Informed RRT*
* Kinodynamic RRT* on Double Integrator
* Kinodynamic Informed RRT* on Double Integrator (with Metropolis-Hastings sampler) 

## Simulations
In this section we present the different commands useful to run the code.
The following command will run the code using the default values, so it will start a simulation of RRT* with a fixed environment and a fixed number of iterations.
```
python main.py
```

##############**!! Exception !!**##############

To run the kinodynamic algorithms (for now) instead, use the following commands:
```
python rrtkino_w_main.py # for kinodynamic rrt*
```
or
```
python informed_rrt_kino_w_main.py #for informed kinodynamic rrt*
```

###############################################

In order to run different configurations the following arguments must be added:
```
-i # it will run the informed version of the rrt* algorithm
-r # it makes the environment randomly generated
-o <number_of_obstacles> # it will specify the number of obstacles randomly generated (the default value is set to 100)
-it <number_if_iterations> # it will specify the number of iterations of the algorithm (default is set to 500)
-c <cost_of_the_best_path> # it will specify the minimum value of the cost of the best path the user want to achieve (if not specified the code will run until reaching the maximum number of iterations)
```

An example of a terminal command to run informed rrt* in a random environment with 80 obstacles and a goal cost of 28 is given below:
```
python main.py -i -r -o 80 -c 28
```
