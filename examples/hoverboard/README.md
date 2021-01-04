## NSGA-II examples ##

The scripts in this directory show examples of using NEAT to control a hoverboard on a game.
It uses Recurrent Networks to control the intensity of both left and right thrusters of the hoverboard, based on it's velocity, angular velocity and normal vector. All those informations could be retreived from real world sensors.

There are two examples:
- __time__: Evolves network with single-fitness _DefaultReproduction_ method, optimizing flight time.
- __timedist__: Evolves network with _NSGA2Reproduction_ method, optimizing two fitness values: flight time and mean squared distance from center.

![hoverboard-reference](https://i.imgur.com/CfrdHmr.gif)

## hoverboard.py

This file implements the game using [pygame](http://pygame.org/).

You can manually play it! However, it's designed to be near impossible without AI (or some USB flight controllers, I guess).

```python
pip install pygame
python hoverboard.py
```

- Q/A : +/- left thruster
- P/L : +/- right thruster

## evolve-time.py

A reference example using a Recurrent Network with the _DefaultReproduction_ method, optimizing a single value of fitness: flightime.
The input values are: velocity (X/Y), angular velocity and normal vector (X/Y).

![hoverboard-reference](https://i.imgur.com/UpJ2HA7.gif)

The evolution converges fast on simple behaviours such as overcoming gravity by boosting both thrusters simultaneously, however a more refined fitness method should include the total variation of velocities and normal vector to help it converge faster to a stable controller.

```
> python evolve-time.py <START_ANGLE>

> python evolve-time.py --help
```

## evolve-timedist.py

A working example using a Recurrent Network with _NSGA2Reproduction_ method, optimizing two fitness values: flight time and mean squared distance from center.
The input values are: velocity (X/Y), angular velocity, normal vector (X/Y) and distance to center (X/Y).

For each genome, instead of a single cycle this method runs 10 game cycles, starting from 5 preset points (including center) with the starting angle A and -A. The fitness results are accumulated and then divided by 10.

![hoverboard-reference](https://i.imgur.com/CfrdHmr.gif)

This method converges a lot faster to results way beyond the convergence point of the default method. More about this at the _Results_ section of this document.

```
> python evolve-timedist.py <START_ANGLE>

> python evolve-timedist.py --help
```

## visualize.py

This is a small tool for viewing the generation data stored at checkpoints.
It allows you to watch the best genome of each generation, as well as plotting fitness and species data over generations.
The plots on the _Results_ section of this document were made with this tool.

```
> python visualize.py <START_ANGLE> <EXPERIMENT>

> python visualize.py --help
```

## gui.py

This is an utilitary lib for drawing neat-python networks using pygame.

# Results

Here's a quick comparison of results found for this particular hoverboard game with and without the use of NSGA-II. These experiments must be improved in order to better outline the benefits and downsides of this approach. Please feel free to develop them further.

The fitness value plotted is Flight Time on both cases. As described above, the NSGA-II example takes the average of 10 runs starting from preset points, to avoid developing behaviours biased on starting at the center.
The observed increase in mean convergence does not seem to rely on these 10 runs evaluation, it is actually harder to evolve in those conditions.

![results_fitness](https://s1.imghub.io/05eik.png)

The distribution of species over generations is heavily affected by NSGA-II. More research is due to evaluate it's cost and benefits. Overall, the species tend to stabilize, having more time to evolve it's features.
A plot of species on the solution space is due to evaluate their distribution, that should be grouped and moving towards the pareto-front.

![results_species](https://s1.imghub.io/05H6H.png)

This plot is messy and needs to be improved. It's a scatter plot of every genome on every generation, color coded.
In order to visualize the overall movement of the population in the solution space, each generation set of points is filled with a Delaunay Triangulation. You can see the generation shapes moving towards the pareto front.
A black line represents the best solution of each generation, so you can see the optimization path and convergence.

![results_pareto](https://s1.imghub.io/05GrJ.png)
