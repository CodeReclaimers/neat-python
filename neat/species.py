# -*- coding: UTF-8 -*-
import random
import sys
from neat.indexer import Indexer
from neat.math_util import mean


class Species(object):
    """ A collection of genetically similar individuals."""
    indexer = Indexer(1)

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
        # choose a new random representative for the species
        self.representative = random.choice(self.members)

    def __str__(self):
        s = "\n   Species {0:2d}   size: {1:3d}   age: {2:3d}   spawn: {3:3d}   ".format(self.ID, len(self.members), self.age, self.spawn_amount)
        s += "\n   No improvement: {0:3d} \t avg. fitness: {1:1.8f}".format(self.no_improvement_age, self.last_avg_fitness)
        return s

    def get_average_fitness(self):
        """ Returns the raw average fitness over all members in the species."""
        return mean([c.fitness for c in self.members])

    def update_stagnation(self):
        """ Updates no_improvement_age based on average fitness progress."""
        fitness = self.get_average_fitness()

        # Check for increase in mean fitness and adjust "no improvement" count as necessary.
        if fitness > self.last_avg_fitness:
            self.last_avg_fitness = fitness
            self.no_improvement_age = 0
        else:
            self.no_improvement_age += 1

    def reproduce(self, config):
        """ Returns a list of 'self.spawn_amount' new individuals """

        offspring = []  # new offspring for this species
        self.age += 1  # increment species age

        self.members.sort()  # sort species's members by their fitness
        self.members.reverse()  # best members first

        if config.elitism:
            # TODO: Wouldn't it be better if we set elitism=2,3,4...
            # depending on the size of each species?
            offspring.append(self.members[0])
            self.spawn_amount -= 1

        # Keep a fraction of the current population for reproduction.
        survivors = int(round(len(self.members) * config.survival_threshold))
        # We always need at least one member for reproduction.
        survivors = max(1, survivors)
        self.members = self.members[:survivors]

        while self.spawn_amount > 0:
            self.spawn_amount -= 1

            # Select two parents at random from the remaining members.
            parent1 = random.choice(self.members)
            parent2 = random.choice(self.members)

            # Note that if the parents are not distinct, crossover should
            # be idempotent. TODO: Write a test for that.
            child = parent1.crossover(parent2)
            offspring.append(child.mutate())

        # reset species (new members will be added again when speciating)
        self.members = []

        # select a new random representative member
        self.representative = random.choice(offspring)

        return offspring
