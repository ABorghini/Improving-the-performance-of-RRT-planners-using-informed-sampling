# Improving-the-performance-of-RRT-planners-using-informed-sampling
<!-- PROJECT SHIELDS -->
[![Python][python-shield]][python-url]
[![Matlab][maltab-shield]][matlab-url]

## Euclidean Domain
|    | RRT* | IRRT* |
| ------------- |:-------:|:------:|
| Random obstacles | ![rand_rrt](https://github.com/ABorghini/Improving-the-performance-of-RRT-planners-using-informed-sampling/assets/87773518/ff2a6c7b-8ede-494f-a5df-dce4e8e30775) | ![rand_irrt](https://github.com/ABorghini/Improving-the-performance-of-RRT-planners-using-informed-sampling/assets/87773518/b8b05cdc-ed47-40fe-88e1-db25e082af42) |
| Square obstacle  | ![quad_rrt_](https://github.com/ABorghini/Improving-the-performance-of-RRT-planners-using-informed-sampling/assets/87773518/a49b9cdc-8fea-4e66-8448-31197a92dae7) | ![quad_irrt_](https://github.com/ABorghini/Improving-the-performance-of-RRT-planners-using-informed-sampling/assets/87773518/ed7510fc-22d6-41bb-adbb-4e9a08f070fd) |

## Kinodynamic Domain
|    | RRTK* | IRRTK* |
| ------------- |:-------:|:------:|
| Fixed environment | ![fixed_rrt](https://github.com/ABorghini/Improving-the-performance-of-RRT-planners-using-informed-sampling/assets/87773518/6f043381-c945-45fc-92ac-a2ef1d175a99) | ![fixed_irrt](https://github.com/ABorghini/Improving-the-performance-of-RRT-planners-using-informed-sampling/assets/87773518/6648c600-fb7a-46cc-af44-4f62f7fda039) | 
| Maze-like environment | ![ml_rrt](https://github.com/ABorghini/Improving-the-performance-of-RRT-planners-using-informed-sampling/assets/87773518/05a9722c-8c4a-417c-8fc7-ebf79a3f3400) | ![ml_irrt](https://github.com/ABorghini/Improving-the-performance-of-RRT-planners-using-informed-sampling/assets/87773518/85d23e7a-2e78-4d27-bb81-47b6a933d0a4) |

## Update
Control over the seeds from the user was added.

All the files were updated to a better plotting.

The implementation of the Kinodynamic RRT* has been improved through matlab engine, now it is faster.
In order to run the kinodynamic implementations a matlab license and matlab engine API are needed.

## About
In this implementation we present the rrt* algorithm in 4 different ways. The implementations are:
* RRT*
* Informed RRT*
* Kinodynamic RRT* on Double Integrator
* Kinodynamic Informed RRT* on Double Integrator (with Metropolis-Hastings sampler) 

## Simulations
In this section we present the different commands useful to run the code.
The following command will run the code using the default values, so it will start a simulation of RRT* with a fixed environment, a fixed number of iterations and a set seed.
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

### Metropolis Hastings `-mh`
```
python main.py -mh
```
Adding this argument will run the euclidean version of the rrt* algorithm with Metropolis Hastings sampler.

### Random `-r`
```
python main.py -r
```
It makes the environment randomly generated.

### Custom environment `-e`
```
python main.py -e
```
It selects the custom environment instead of the fixed one.

### Sqaure environment `-q`
```
python main.py -q
```
It selects the square environment.

### Obstacles number `-o`
```
python main.py -r -o <number_of_obstacles>
```
It will specify the number of obstacles to generate if the environment is random. (the default value is set to 100)

### Iterations number `-it`
```
python main.py -it <number_if_iterations>
``` 
It will specify the number of iterations of the algorithm (default is set to 500).

### Goal path cost `-c`
```
python main.py -c <cost_of_the_best_path>
```
### Fix near radius `-fnr`
```
python main.py -fnr
```
Set the radius close to a fixed default value instead of iteratively decreasing it during the execution.

It will specify the minimum value of the cost of the best path the user want to achieve (if not specified the code will run until reaching the maximum number of iterations).

After different simulations we found out suitable values to choose for this argument. (Differences are mainly due to the environment)
* euclidean version, random environment: select a number between 11 and 20
* euclidean version, fixed environment: select a number between 24 and 27
* euclidean version, custom environment: select a number between 170 and 180
* kinodynamic version, fixed environment: select a number between 90 and 100
* kinodynamic version, custom environment: select a number between 170 and 180

### Process seed `-s`
```
python main.py -s <seed_value>
```
It will specify the seed of the process, that controls the randomness in the generation of the next nodes, and so of the generated path.

### Environment seed `-es`
```
python main.py -es <environment_seed_value>
```
It will specify the seed of the environment, that controls the random distribution of the obstacles. It has effect just in the random environment.

### Examples
An example of a terminal command to run informed rrt* in a random environment with 80 obstacles and a goal cost of 28 is given below:
```
python main.py -i -r -o 80 -c 28
```
Another example, it will run kinodynamic rrt* in a fixed environment with a maximum number of iterations equal to 350
```
python main.py -k -it 350
```
<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[python-shield]: https://img.shields.io/badge/python-v3.10-brightgreen
[python-url]: https://www.python.org/downloads/release/python-3100/
[maltab-shield]: https://img.shields.io/badge/matlab-v2023a-brightgreen
[matlab-url]: https://it.mathworks.com/products/matlab
