# NSGA-II + NEAT

[NSGA-II](https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf) (Non-Dominated Sorting Genetic Algorithm) is en Elitist Multiobjective Genetic Algorithm, designed to
efficiently sort and select individuals on a population based on multiple fitness values.

## Overview

This is an implementation of NSGA-II as a reproduction method for NEAT.
The `DefaultReproduction` implements a simple single-fitness species-wise tournament with a fixed, user-specified, elitism and survival ratio.
`NSGA2Reproduction` has a few additional steps for sorting the population based on multiple fitness values. The last population is always stored and compared with the new one, to ensure parameterless elitism.

- Speciation is a novel concept to _NSGA-II_, so the design decision was to sort the merged population with no constraints to species, then pick the overall best genomes. This allows species to grow and shrink based on their elites. The tournament then offsprings the current size of each species.
This seems to keep species more stable over the generations, while also improving their mean convergence. However more studies are required to validate these results.

- Stagnation is also a novel concept to _NSGA-II_, and the current design decision is to stagnate species of the child population before merging it with the parent one. This allows species that are doing fine to slowly stagnate, instead of going extinct in a single generation. It is still not clear how this impacts the overall behaviour of species.

Below is an schematic of the reproduction method.
It's described following the [original article](https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf) Main Loop, on pages 185 and 186.

![nsga-ii concept](https://raw.githubusercontent.com/hugoaboud/neat-python/a4c90ad777439482831a7c5c6067899e64805a92/neat/nsga2/nsga2neat.svg)

- The current implementation does not differentiate the first generation, it runs the second procedure with an empty parent population, so the results are the same as the proposed first step;
- Please note that the sorting results on the picture might not be accurate;
- A small rectangle with the text "best_genome" indicates where the best genome is evaluated. Note that it must happen before tournament, but after merging populations, to ensure elitism;
- Stagnation is not illustrated on the schematic, but it's the first thing on the `sort()` method, it removes genomes from child population Q(t);

You can find a working documented example in [/examples/hoverboard/](https://github.com/hugoaboud/neat-python/tree/master/examples/hoverboard).

## Implementation Notes

- In order to avoid unecessary changes to the neat-python library, a class named NSGA2Fitness was created. It overloads the operators used by the lib;
- In order to comply with the single fitness progress/threshold, the first fitness value is used for thresholding and when the fitness object is converted to a float (like in math_util.mean);
- In order to use the multiobjective crowded-comparison operator, fitness functions config should always be set to 'max'; 
- Front ranks are negative, so picking the best becomes a maximization problem, the default behaviour of neat-python;

# Implementation

1. A _NSGA2Fitness_ class is used to store multiple fitness values for each genome during evaluation (eval_genomes);
2. After all child genomes _Q(t)_ are evaluated, _NSGA2Reproduction.sort()_ method is run by _Population_;
3. _sort()_ starts by removing stagnated genomes and species from the child population _Q(t)_;
4. _sort()_ then merges the child _Q(t)_ and parent _P(t)_ population (saved from the last generation), and sorts it in parento-fronts; inside each pareto-front, genomes are sorted in decreasing order with the crowding-distance operator; if two values from the same front share the same crowding-distance, the decision relies on the first fitness value;
5. The new parent population _P(t+1)_ is selected from the best fronts, and it's species are saved for later merging (in case they go extinct);
6. The best genome is picked up from the new parent population _P(t+1)_;
7. The first fitness value is used to check for fitness threshold and average fitness calculation;
8. _reproduce()_ is then called by _Population_, and the genome tournament (selection, crossover, mutation) creates the new child population _Q(t+1)_;
9. Tournament is done species-wise, for the reasons outlined by the NEAT algorithm (avoid genetic aberrations, preserve innovation);
10. The resulting population _Q(t+1)_ will be evaluated on the next iteration;
