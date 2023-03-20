# Improving-the-performance-of-RRT-planners-using-informed-sampling

## Update
The informed version of the kinodynamic RRT* has been uploaded.

All the files were updated to a better plotting.

The implementation of the Kinodynamic RRT* has been improved, now it is faster.

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

In order to run different configurations the following parameters must be set:
### Informed `-i`
```
python main.py -i
```

Adding this argument will run the informed version of the rrt* algorithm.
### Kinodynamic `-k`
```
python main.py -k
```
Adding this argument will run the kinodynamic version of the rrt* algorithm.
### Random `-r`
```
python main.py -r 
```
It makes the environment randomly generated.
### Obstacles number `-o`
```
python main.py -r -o <number_of_obstacles>
```
It will specify the number of obstacles to generate if the environment is random. (the default value is set to 100)
### Iterations number `-it`
```
python main.py -it <number_if_iterations>
``` 
It will specify the number of iterations of the algorithm (default is set to 50).

Due to implementation differences between the kinodynamic and the geometric versions, the suiting number of iterations differs.
* euclidean version: select a number between 500 and 5000
* kinodynamic version: select a number between 50 and 500
### Goal path cost `-c`
```
python main.py -c <cost_of_the_best_path>
```
It will specify the minimum value of the cost of the best path the user want to achieve (if not specified the code will run until reaching the maximum number of iterations).

After different simulations we found out suitable values to choose for this argument. (Differences are mainly due to the environment)
* euclidean version: select a number between 25 and 30
* kinodynamic version: select a number between 40 and 65 (default goal is [50, 65, 0, 0] to guarantee a faster solution; choose between 90 and 110 if the goal is set to [98, 98, 0, 0], for a complete search)

### Examples
An example of a terminal command to run informed rrt* in a random environment with 80 obstacles and a goal cost of 28 is given below:
```
python main.py -i -r -o 80 -c 28
```
Another example, it will run kinodynamic rrt* in a fixed environment with a maximum number of iterations equal to 350
```
python main.py -k -it 350
```
