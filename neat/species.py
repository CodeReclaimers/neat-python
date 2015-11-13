# -*- coding: UTF-8 -*-
import random
from neat.math_util import mean


class Species(object):
    """ A subpopulation containing similar individiduals """
    __next_id = 1  # global species id counter

    @classmethod
    def __get_next_id(cls, previous_id):
        if previous_id is None:
            previous_id = cls.__next_id
            cls.__next_id += 1

        return previous_id

    def __init__(self, first_individual, previous_id=None):
        """ A species requires at least one individual to come to existence """
        self.ID = self.__get_next_id(previous_id)  # species's id
        self.age = 0  # species's age
        self.members = []  # species's individuals
        self.add(first_individual)
        self.hasBest = False  # Does this species has the best individual of the population?
        self.spawn_amount = 0

        # The number of generations since the mean fitness increased.
        self.no_improvement_age = 0

        self.last_avg_fitness = 0

        self.representative = first_individual

    def add(self, individual):
        """ Add a new individual to the species """
        # set individual's species id
        individual.species_id = self.ID
        # add new individual
        self.members.append(individual)
        # choose a new random representative for the species
        self.representative = random.choice(self.members)

    def __str__(self):
        s = "\n   Species %2d   size: %3d   age: %3d   spawn: %3d   " \
            % (self.ID, len(self), self.age, self.spawn_amount)
        s += "\n   No improvement: %3d \t avg. fitness: %1.8f" \
             % (self.no_improvement_age, self.last_avg_fitness)
        return s

    def tournament_selection(self, k=2):
        """ Tournament selection with size k (default k=2).
            Make sure the population has at least k individuals """
        random.shuffle(self.members)

        return max(self.members[:k])

    def average_fitness(self):
        """ Returns the raw average fitness for this species """
        current = mean([c.fitness for c in self.members])

        # Check for increase in mean fitness and adjust "no improvement" count as necessary.
        if current > self.last_avg_fitness:
            self.last_avg_fitness = current
            self.no_improvement_age = 0
        else:
            self.no_improvement_age += 1

        return current

    def reproduce(self, config):
        """ Returns a list of 'spawn_amount' new individuals """

        offspring = []  # new offspring for this species
        self.age += 1  # increment species age

        # print "Reproducing species %d with %d members" %(self.id, len(self.members))

        # this condition is useless since no species with spawn_amount < 0 will
        # reach this point - at least it shouldn't happen.
        # assert self.spawn_amount > 0, "Species %d with zero spawn amount!" % (self.ID)

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

            if len(self.members) > 1:
                # Selects two parents from the remaining species and produces a single individual
                # Stanley selects at random, here we use tournament selection (although it is not
                # clear if has any advantages)
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()

                assert parent1.species_id == parent2.species_id, "Parents has different species id."
                child = parent1.crossover(parent2)
                offspring.append(child.mutate())
            else:
                # mutate only
                parent1 = self.members[0]
                # TODO: temporary hack - the child needs a new id (not the father's)
                child = parent1.crossover(parent1)
                offspring.append(child.mutate())

        # reset species (new members will be added again when speciating)
        self.members = []

        # select a new random representative member
        self.representative = random.choice(offspring)

        return offspring
