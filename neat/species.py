# -*- coding: UTF-8 -*-
import random
from config import Config

class Species(object):
    """ A subpopulation containing similar individiduals """
    __id = 0 # global species id counter

    def __init__(self, first_individual, previous_id=None):
        """ A species requires at least one individual to come to existence """
        self.__id = self.__get_new_id(previous_id)    # species's id
        self.__age = 0                                # species's age
        self.__subpopulation = []                     # species's individuals
        self.add(first_individual)
        self.hasBest = False                        # Does this species has the best individual of the population?
        self.spawn_amount = 0
        self.no_improvement_age = 0                 # the age species has shown no improvements on average

        self.__last_avg_fitness = 0

        self.representant = first_individual

    members = property(lambda self: self.__subpopulation)
    age = property(lambda self: self.__age)
    id  = property(lambda self: self.__id)

    @classmethod
    def __get_new_id(cls, previous_id):
        if previous_id is None:
            cls.__id += 1
            return cls.__id
        else:
            return previous_id

    def add(self, individual):
        """ Add a new individual to the species """
        # set individual's species id
        individual.species_id = self.__id
        # add new individual
        self.__subpopulation.append(individual)
        # choose a new random representant for the species
        self.representant = random.choice(self.__subpopulation)

    def __iter__(self):
        """ Iterates over individuals """
        return iter(self.__subpopulation)

    def __len__(self):
        """ Returns the total number of individuals in this species """
        return len(self.__subpopulation)

    def __str__(self):
        s  = "\n   Species %2d   size: %3d   age: %3d   spawn: %3d   " \
                %(self.__id, len(self), self.__age, self.spawn_amount)
        s += "\n   No improvement: %3d \t avg. fitness: %1.8f" \
                %(self.no_improvement_age, self.__last_avg_fitness)
        return s

    def TournamentSelection(self, k=2):
        """ Tournament selection with size k (default k=2).
            Make sure the population has at least k individuals """
        random.shuffle(self.__subpopulation)

        return max(self.__subpopulation[:k])

    def average_fitness(self):
        """ Returns the raw average fitness for this species """
        sum = 0.0
        for c in self.__subpopulation:
            sum += c.fitness

        try:
            current = sum/len(self)
        except ZeroDivisionError:
            print "Species %d, with length %d is empty! Why? " % (self.__id, len(self))
        else:  # controls species no improvement age
            # if no_improvement_age > threshold, species will be removed
            if current > self.__last_avg_fitness:
                self.__last_avg_fitness = current
                self.no_improvement_age = 0
            else:
                self.no_improvement_age += 1

            return current

    def reproduce(self):
        """ Returns a list of 'spawn_amount' new individuals """

        offspring = [] # new offspring for this species
        self.__age += 1  # increment species age

        #print "Reproducing species %d with %d members" %(self.id, len(self.__subpopulation))

        # this condition is useless since no species with spawn_amount < 0 will
        # reach this point - at least it shouldn't happen.
        #assert self.spawn_amount > 0, "Species %d with zero spawn amount!" % (self.__id)

        self.__subpopulation.sort()     # sort species's members by their fitness
        self.__subpopulation.reverse()  # best members first

        if Config.elitism:
            # TODO: Wouldn't it be better if we set elitism=2,3,4...
            # depending on the size of each species?
            offspring.append(self.__subpopulation[0])
            self.spawn_amount -= 1

        survivors = int(round(len(self)*Config.survival_threshold)) # keep a % of the best individuals

        if survivors > 0:
            self.__subpopulation = self.__subpopulation[:survivors]
        else:
            # ensure that we have at least one chromosome to reproduce
            self.__subpopulation = self.__subpopulation[:1]

        while(self.spawn_amount > 0):

            self.spawn_amount -= 1

            if len(self) > 1:
                # Selects two parents from the remaining species and produces a single individual
                # Stanley selects at random, here we use tournament selection (although it is not
                # clear if has any advantages)
                parent1 = self.TournamentSelection()
                parent2 = self.TournamentSelection()

                assert parent1.species_id == parent2.species_id, "Parents has different species id."
                child = parent1.crossover(parent2)
                offspring.append(child.mutate())
            else:
                # mutate only
                parent1 = self.__subpopulation[0]
                # TODO: temporary hack - the child needs a new id (not the father's)
                child = parent1.crossover(parent1)
                offspring.append(child.mutate())

        # reset species (new members will be added again when speciating)
        self.__subpopulation = []

        # select a new random representant member
        self.representant = random.choice(offspring)

        return offspring
