"""

    Implementation of NSGA-II as a reproduction method for NEAT.
    More details on the README.md file.

    @autor: Hugo Aboud (@hugoaboud)

"""
from __future__ import division

import math
import random
from itertools import count
from operator import add

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean
from neat.species import Species

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
    def sort(self, population, species, pop_size, generation):

        ## Stagnation happens only on the child Q(t) population, before merging,
        # so the species have a chance to avoid stagnation if they're doing
        # generally fine
        # Filter out stagnated species genomes, collect the set of non-stagnated
        remaining_species = {} # remaining species
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            # stagnant species: remove genomes from child population
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
                population = {id:g for id,g in population.items() if g not in stag_s.members}
            # non stagnant species: append species to parent species dictionary
            else:
                remaining_species[stag_sid] = stag_s

        # No genomes left.
        if not remaining_species:
            species.species = {}
            return {}

        ## NSGA-II : step 1 : merge and sort
        # Merge populations P(t)+Q(t) and sort by non-dominated fronts
        child_pop = [g for _, g in population.items()] + self.parent_pop

        # Merge parent P(t) species and child (Qt) species,
        # so all non-stagnated genomes are covered by species.species
        species.species = remaining_species
        for id, sp in self.parent_species.items():
            if (id in species.species):
                species.species[id].members.update(sp.members)
            else:
                species.species[id] = sp

        ## Non-Dominated Sorting (of P(t)+Q(t))
        # algorithm data
        S = {} # genomes dominated by key genome
        n = {} # counter of genomes dominating key genome
        F = [] # current dominance front
        self.fronts = [] # clear dominance fronts
        # calculate dominance of every genome to every other genome - O(MN²)
        for p in range(len(child_pop)):
            S[p] = []
            n[p] = 0
            for q in range(len(child_pop)):
                if (p == q): continue
                # p dominates q
                if (child_pop[p].fitness.dominates(child_pop[q].fitness)):
                    S[p].append(q)
                # q dominates p
                elif (child_pop[q].fitness.dominates(child_pop[p].fitness)):
                    n[p] += 1
            # if genome is non-dominated, set rank and add to front
            if (n[p] == 0):
                child_pop[p].fitness.rank = 0
                F.append(p)

        # assemble dominance fronts - O(N²)
        i = 0 # dominance front iterator
        while (len(F) > 0):
            # store front
            self.fronts.append([child_pop[f] for f in F])
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
                        child_pop[q].fitness.rank = -(i+1)
                        Q.append(q)
            # iterate front
            i += 1
            F = Q

        ## NSGA-II : step 2 : pareto selection
        # Create new parent population P(t+1) from the best fronts
        # Sort each front by Crowding Distance, to be used on Tournament
        self.parent_pop = []
        for front in self.fronts:
            ## Calculate crowd-distance of fitnesses
            # First set distance to zero
            for genome in front:
                genome.dist = 0
            # List of fitnesses to be used for distance calculation
            fitnesses = [f.fitness for f in front]
            # Iterate each fitness parameter (values)
            for m in range(len(fitnesses[0].values)):
                # Sort fitnesses by parameter
                fitnesses.sort(key=lambda f: f.values[m])
                # Get scale for normalizing values
                scale = (fitnesses[-1].values[m]-fitnesses[0].values[m])
                # Set edges distance to infinite, to ensure are picked by the next step
                # This helps keeping the population diverse
                fitnesses[0].dist = float('inf')
                fitnesses[-1].dist = float('inf')
                # Increment distance values for each fitness
                if (scale > 0):
                    for i in range(1,len(fitnesses)-1):
                        fitnesses[i].dist += abs(fitnesses[i+1].values[0]-fitnesses[i-1].values[0])/scale

            ## Sort front by crowd distance
            # In case distances are equal (mostly on 'inf' values), use the first value to sort
            front.sort(key=lambda g: (g.fitness.dist,g.fitness.values[0]), reverse = True)

            ## Assemble new parent population P(t+1)
            # front fits entirely on the parent population, just append it
            if (len(self.parent_pop) + len(front) <= pop_size):
                self.parent_pop += front
                if (len(self.parent_pop) == pop_size): break
            # front exceeds parent population, append only what's necessary to reach pop_size
            else:
                self.parent_pop += front[:pop_size-len(self.parent_pop)]
                break


        ## NSGA-II : post step 2 : Clean Species
        # Remove the genomes that haven't passed the crowding-distance step
        # (The ones stagnated are already not on this dict)
        # Also rebuild SpeciesSet.genome_to_species
        species.genome_to_species = {}
        for _, sp in species.species.items():
            sp.members = {id:g for id,g in sp.members.items() if g in self.parent_pop}
            # map genome to species
            for id, g in sp.members.items():
                species.genome_to_species[id] = sp.key
        # Remove empty species
        species.species = {id:sp for id,sp in species.species.items() if len(sp.members) > 0}

        # self.parent_species should be a deepcopy of the species dictionary,
        # in order to avoid being modified by the species.speciate() method
        # the species in here are used to keep track of parent_genomes on next sort
        self.parent_species = {}
        for id, sp in species.species.items():
            self.parent_species[id] = Species(id, sp.created)
            self.parent_species[id].members = dict(sp.members)
            self.parent_species[id].representative = sp.representative

        ## NSGA-II : end : return parent population P(t+1) to be assigned to child population container Q(t+1)
        # this container will be used on the Tournament at NSGA2Reproduction.reproduce()
        # to create the real Q(t+1) population
        return {g.key:g for g in self.parent_pop}

    # NSGA-II step 2: crowding distance sorting
    # this is where NSGA-2 reproduces the population by the fitness rank
    # calculated on step 1
    def reproduce(self, config, species, pop_size, generation):

        ## NSGA-II : step 3 : Tournament
        # Disclaimer: this method uses no absolute fitness values
        # The fitnesses are compared through the crowded-comparison operator
        # fitness.values[0] is used for fitness threshold and reporting, but not in here

        ## Tournament
        # Each species remains the same size (they grow and shrink based on pareto-fronts, on sort())
        # Only the <survival_threshold> best are used for mating
        # Mating can be sexual or asexual
        new_population = {}
        for _, sp in species.species.items():
            # Sort species members by crowd distance
            members = list(sp.members.values())
            members.sort(key=lambda g: g.fitness, reverse=True)
            # Survival threshold: how many members should be used as parents
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold * len(members)))
            # Use at least two parents no matter what the threshold fraction result is.
            members = members[:max(repro_cutoff, 2)]
            # spawn the number of members on the species
            spawn = len(sp.members)
            for i in range(spawn):
                # pick two random parents
                parent_a = random.choice(members)
                parent_b = random.choice(members)
                # sexual reproduction
                # if a == b, it's asexual reproduction
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent_a, parent_b, config.genome_config)
                child.mutate(config.genome_config)
                new_population[gid] = child

        return new_population
