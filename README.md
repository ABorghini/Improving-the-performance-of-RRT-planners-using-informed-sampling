# Improving-the-performance-of-RRT-planners-using-informed-sampling

## About
In this implementation we present the rrt* algorithm in 4 different ways. The implementations are:
* RRT*
* Informed RRT*
* Kynodinamic RRT* on Double Integrator
* Kynodinamic Informed RRT* on Double Integrator (with Metropolis-Hastings sampler) 

## Simulations
In this section we present the different commands useful to run the code.
The following command will run the code using the default values, so it will start a simulation of RRT* with a fixed environment and a fixed number of iterations.
```
python main.py
```
In order to run different configurations the following arguments must be added:
```
-i # it will run the informed version of the rrt* algorithm
-r # it makes the environment randomly generated
-o <number_of_obstacles> # it will specify the number of obstacles randomly generated (the default value is set to 100)
-it <number_if_iterations> # it will specify the number of iterations of the algorithm (default is set to 500)
-c <cost_of_the_best_path> # it will specify the minimum value of the cost of the best path the user want to achieve (the default value is 0.0, in this way the algorithm will stop when it reaches the max number of iterations)
```
