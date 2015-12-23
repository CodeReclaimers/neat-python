# -*- coding: UTF-8 -*-
import math
import random
import sys
from neat.indexer import Indexer
from neat.math_util import mean


class Species(object):
    """ A collection of genetically similar individuals."""
    indexer = Indexer(1)

    @classmethod
    def clear_indexer(cls):
        cls.indexer.clear()

    def __init__(self, first_individual, previous_id=None):
        self.representative = first_individual
        self.ID = Species.indexer.next(previous_id)
        self.age = 0
        self.members = []
        self.add(first_individual)
        self.spawn_amount = 0
        self.last_avg_fitness = -sys.float_info.max
        self.no_improvement_age = 0

    def add(self, individual):
        individual.species_id = self.ID
        self.members.append(individual)

    def get_average_fitness(self):
        """ Returns the average fitness over all members in the species."""
        return mean([c.fitness for c in self.members])

    def update_stagnation(self):
        """ Updates no_improvement_age based on average fitness progress."""
        fitness = self.get_average_fitness()
        if fitness > self.last_avg_fitness:
            self.last_avg_fitness = fitness
            self.no_improvement_age = 0
        else:
            self.no_improvement_age += 1

    def reproduce(self, config):
        """
        Update species age, clear the current membership list, and return a list of 'self.spawn_amount' new individuals.
        """
        self.age += 1

        # Sort with most fit members first.
        self.members.sort(reverse=True)

        offspring = []
        if config.elitism > 0:
            offspring.extend(self.members[:config.elitism])
            self.spawn_amount -= config.elitism

        # Keep a fraction of the current population for reproduction.
        survivors = int(math.ceil(len(self.members) * config.survival_threshold))
        # We always need at least one member for reproduction.
        survivors = max(1, survivors)
        self.members = self.members[:survivors]

        while self.spawn_amount > 0:
            self.spawn_amount -= 1

            # Select two parents at random from the given set of members.
            parent1 = random.choice(self.members)
            parent2 = random.choice(self.members)

            # Note that if the parents are not distinct, crossover should produce a
            # genetically identical clone of the parent (but with a different ID).
            child = parent1.crossover(parent2)
            offspring.append(child.mutate())

        # Reset species members--the speciation process in Population will repopulate this list.
        self.members = []

        # Select a new random representative member from the new offspring, and remove
        # the representative from the list (so that each species always gets at least one member).
        self.representative = random.choice(offspring)
        self.add(self.representative)
        offspring.remove(self.representative)

        return offspring
