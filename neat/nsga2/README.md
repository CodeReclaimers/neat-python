# NSGA-II + NEAT



### Overview

NSGA-II (Non-Dominated Sorting Genetic Algorithm) is en Elitist Multiobjective Genetic Algorithm, designed to
efficiently sort populations based on multiple fitness values.

The algorithm is proposed in two steps:
    - 1: Fast Non-dominated Sorting
    - 2: Crowding Distance Sorting

Step 1 sorts the population in Parento-Front groups.
Step 2 creates a new population from the sorted old one

# IMPLEMENTATION NOTES

- In order to avoid unecessary changes to the neat-python library, a class
named NSGA2Fitness was created. It overloads the operators used by the lib,
keeping it concise with the definition.
- In order to comply with the single fitness progress/threshold, the first
fitness value is used for thresholding and when it's converted to a float
(like in mean methods).
- In order to use the multiobjective crowded-comparison operator, fitness
functions config should always be set to 'max'.
- Ranks are negative, so it's a maximization problem, as the default examples

# IMPLEMENTATION

- A NSGA2Fitness class is used to store multiple fitness values
  during evaluation
- NSGA2Reproduction keeps track of parent population and species
- After all new genomes are evaluated, sort() method must be run
- sort() merges the current and parent population and sorts it
  in parento-fronts, assigning a rank value to each
- When reproduce() is called by population, the default species
  stagnation runs
- Then, Crowding Distance Sorting is used to remove the worst
  genomes from the remaining species.
- The best <pop_size> genomes are stored as the parent population
- Each species then doubles in size by sexual/asexual reproduction
- TODO: If pop_size was not reached, cross genomes from different fronts
  to incentivize innovation
