"""

    Implementation of NSGA-II as a reproduction method for NEAT.

    @autor: Hugo Aboud (@hugoaboud)

    # OVERVIEW

    NSGA-II is en Elitist Multiobjective Genetic Algorithm, designed to
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

"""
from __future__ import division

import math
import random
from itertools import count
from operator import add

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean

##
#   NSGA-II Fitness
#   Stores multiple fitness values
#   Overloads operators allowing integration to unmodified neat-python
##

class NSGA2Fitness:
    def __init__(self, *values):
        self.values = values
        self.rank = 0
        self.dist = 0.0
        #self.score = 0.0
    def set(self, *values):
        self.values = values
    def add(self, *values):
        self.values = list(map(add, self.values, values))

    def dominates(self, other):
        d = False
        for a, b in zip(self.values, other.values):
            if (a < b): return False
            elif (a > b): d = True
        return d

    # >
    def __gt__(self, other):
        # comparison of fitnesses on tournament, use crowded-comparison operator
        # this is also used by max/min
        if (isinstance(other,NSGA2Fitness)):
            if (self.rank > other.rank): return True
            elif (self.rank == other.rank and self.dist > other.dist): return True
            return False
        # stagnation.py initializes fitness as -sys.float_info.max
        # it's the only place where the next line should be called
        return self.rank > other
    # >=
    def __ge__(self, other):
        # population.run() compares fitness to the fitness threshold for termination
        # it's the only place where the next line should be called
        # it's also the only place where score participates of evolution
        # besides that, score is a value for reporting the general evolution
        return self.values[0] >= other
    # -
    def __sub__(self, other):
        # used only by reporting->neat.math_util to calculate fitness (score) variance
        #return self.score - other
        return self.values[0] - other
    # float()
    def __float__(self):
        # used only by reporting->neat.math_util to calculate mean fitness (score)
        #return self.score
        return float(self.values[0])
    # str()
    def __str__(self):
        #return "rank:{0},score:{1},values:{2}".format(self.rank, self.score, self.values)
        return "rank:{0},dist:{1},values:{2}".format(self.rank, self.dist, self.values)

##
#   NSGA-II Reproduction
#   Implements "Non-Dominated Sorting" and "Crowding Distance Sorting" to reproduce the population
##

class NSGA2Reproduction(DefaultClassConfig):
    @classmethod
    def parse_config(cls, param_dict):

        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2)])

    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation

        # Parent population and species
        # This population is mixed with the evaluated population in order to achieve elitism
        self.parent_pop = []
        self.parent_species = {}

        # Parento-fronts of genomes (including population and parent population)
        # These are created by the sort() method at the end of the fitness evaluation process
        self.fronts = []

    # new population, called by the population constructor
    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
        return new_genomes

    # NSGA-II step 1: fast non-dominated sorting
    # This >must< be called by the fitness function (aka eval_genomes)
    # after a NSGA2Fitness was assigned to each genome
    def sort(self, genomes):
        print("NSGA-II step 1: non-dominated sorting")
        genomes = [g[1] for g in genomes] + self.parent_pop
        # algorithm data
        S = {} # genomes dominated by key genome
        n = {} # counter of genomes dominating key genome
        F = [] # current dominance front
        self.fronts = [] # clear dominance fronts
        # calculate dominance of every genome to every other genome - O(MN²)
        for p in range(len(genomes)):
            S[p] = []
            n[p] = 0
            for q in range(len(genomes)):
                if (p == q): continue
                # p dominates q
                if (genomes[p].fitness.dominates(genomes[q].fitness)):
                    S[p].append(q)
                # q dominates p
                elif (genomes[q].fitness.dominates(genomes[p].fitness)):
                    n[p] += 1
            # if genome is non-dominated, set rank and add to front
            if (n[p] == 0):
                genomes[p].fitness.rank = 0
                F.append(p)

        # assemble dominance fronts - O(N²)
        i = 0 # dominance front iterator
        while (len(F) > 0):
            # store front
            self.fronts.append([genomes[f] for f in F])
            # new dominance front
            Q = []
            # for each genome in current front
            for p in F:
                # for each genome q dominated by p
                for q in S[p]:
                    # decrease dominate counter of q
                    n[q] -= 1
                    # if q reached new front
                    if n[q] == 0:
                        genomes[q].fitness.rank = -(i+1)
                        Q.append(q)
            # iterate front
            i += 1
            F = Q

    # NSGA-II step 2: crowding distance sorting
    # this is where NSGA-2 reproduces the population by the fitness rank
    # calculated on step 1
    def reproduce(self, config, species, pop_size, generation):

        # Disclaimer: this method uses no absolute fitness values
        # The fitnesses are compared through the crowded-comparison operator
        # fitness.values[0] is used for fitness threshold and reporting, but not in here
        print("NSGA-II step 2: crowding distance sorting")

        # append parent species to list, so all front genomes are covered
        species.species.update(self.parent_species)

        # Default Stagnation without fitness calculation
        # Filter out stagnated species genomes, collect the set of non-stagnated
        remaining_species = {}
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                remaining_species[stag_sid] = stag_s

        # No genomes left.
        if not remaining_species:
            species.species = {}
            return {}

        # Crowding distance assignment
        # Create new parent population from the best fronts
        self.parent_pop = []
        for front in self.fronts:

            ## WIP: Calculate crowd-distance
            for genome in front:
                genome.dist = 0
            fitnesses = [f.fitness for f in front]
            for m in range(len(fitnesses[0].values)):
                fitnesses.sort(key=lambda f: f.values[m])
                scale = (fitnesses[-1].values[m]-fitnesses[0].values[m])
                fitnesses[0].dist = float('inf')
                fitnesses[-1].dist = float('inf')
                if (scale > 0):
                    for i in range(1,len(fitnesses)-1):
                        fitnesses[i].dist += abs(fitnesses[i+1].values[0]-fitnesses[i-1].values[0])/scale

            # front fits entirely on the parent population, just append it
            if (len(self.parent_pop) + len(front) < pop_size):
                self.parent_pop += front
            # front exceeds parent population, sort by crowd distance and
            # append only what's necessary to reach pop_size
            else:
                front.sort(key=lambda g: g.fitness)
                self.parent_pop += front[:pop_size-len(self.parent_pop)]

        # Map parent species, by removing the genomes from remaining_species
        # that haven't passed the crowding-distance step
        self.parent_species = remaining_species
        for _, sp in remaining_species.items():
            # filter genomes from each species
            sp.member = [g for g in sp.members if g in self.parent_pop]

        # Remove empty species
        self.parent_species = {id:sp for id,sp in self.parent_species.items() if len(sp.members) > 0}

        # Reproduce species of parent population into new population
        # Each species doubles in size
        new_population = {}
        for _, species in self.parent_species.items():
            # spawn the number of members on the species
            spawn = len(species.members)
            # special case: single member, asexual reproduction
            if (spawn == 1):
                parent = [g for _, g in species.members.items()][0]
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent, parent, config.genome_config)
                child.mutate(config.genome_config)
                new_population[gid] = child
            # usual case: n > 1 members, sexual reproduction
            else:
                for i in range(spawn):
                    # pick two random parents
                    parents = random.sample(list(species.members.values()), 2)
                    # sexual reproduction
                    gid = next(self.genome_indexer)
                    child = config.genome_type(gid)
                    child.configure_crossover(parents[0], parents[1], config.genome_config)
                    child.mutate(config.genome_config)
                    new_population[gid] = child

        return new_population
