from __future__ import print_function

import copy
import gzip
import random
import time
import pickle

from neat.config import Config
from neat.genome import Genome, FFGenome
from neat.genes import NodeGene, ConnectionGene
from neat.species import Species
from neat.math_util import mean, stdev
from neat.diversity import AgedFitnessSharing


class Population(object):
    """ Manages all the species  """

    def __init__(self, config, checkpoint_file=None, initial_population=None,
                 node_gene_type=NodeGene, conn_gene_type=ConnectionGene,
                 diversity_type=AgedFitnessSharing):

        # If config is not a Config object, assume it is a path to the config file.
        if not isinstance(config, Config):
            config = Config(config)

        self.config = config

        self.population = None
        self.node_gene_type = node_gene_type
        self.conn_gene_type = conn_gene_type
        self.diversity = diversity_type(self.config)

        if checkpoint_file:
            # Start from a saved checkpoint.
            self.__resume_checkpoint(checkpoint_file)
        else:
            # currently living species
            self.__species = []
            # species history
            self.species_log = []
            self.species_fitness_log = []

            # List of statistics for all generations.
            self.avg_fitness_scores = []
            self.most_fit_genomes = []

            if initial_population is None:
                self.__create_population()
            else:
                self.population = initial_population
            self.generation = -1

    def __resume_checkpoint(self, checkpoint):
        '''
        Resumes the simulation from a previous saved point. This is done by swapping out our existing
        __dict__ with the loaded population's.
        '''
        # TODO: Wouldn't it just be better to create a class method to load and return the stored Population
        # object as-is?  I don't know if there are hidden side effects to directly replacing __dict__.
        with gzip.open(checkpoint) as f:
            print('Resuming from a previous point: {0!s}'.format(checkpoint))
            # when unpickling __init__ is not called again
            previous_pop = pickle.load(f)
            self.__dict__ = previous_pop.__dict__

            print('Loading random state')
            random.setstate(pickle.load(f))

    def __create_checkpoint(self, report):
        """ Saves the current simulation state. """
        if report:
            print('Creating checkpoint file at generation: {0:d}'.format(self.generation))

        with gzip.open('checkpoint_' + str(self.generation), 'w', compresslevel=5) as f:
            # Write the entire population state.
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            # Remember the current random number state.
            pickle.dump(random.getstate(), f, protocol=2)

    def __create_population(self):
        if self.config.feedforward:
            genotypes = FFGenome
        else:
            genotypes = Genome

        self.population = []
        # TODO: Add FS-NEAT support, which creates an empty connection set, and then performs a
        # single add connection mutation.
        # This would give three initialization methods:
        # 1. Fully connected (each input connected to all outputs)
        # 2. "minimally" connected, which isn't really minimal (one random input to each output)
        # 3. FS-NEAT connected (one random connection)
        if self.config.fully_connected:
            for i in range(self.config.pop_size):
                g = genotypes.create_fully_connected(self.config, self.node_gene_type, self.conn_gene_type)
                self.population.append(g)
        else:
            for i in range(self.config.pop_size):
                g = genotypes.create_minimally_connected(self.config, self.node_gene_type, self.conn_gene_type)
                self.population.append(g)

        if self.config.hidden_nodes > 0:
            for g in self.population:
                g.add_hidden_nodes(self.config.hidden_nodes)

    def __repr__(self):
        s = "Population size: {0:d}".format(self.config.pop_size)
        s += "\nTotal species: {0:d}".format(len(self.__species))
        return s

    def __speciate(self, report):
        """ Group genomes into species by similarity """
        # Speciate the population
        for individual in self.population:
            # Find the species with the most similar representative.
            min_distance = None
            closest_species = None
            for s in self.__species:
                distance = individual.distance(s.representative)
                if distance < self.config.compatibility_threshold:
                    if min_distance is None or distance < min_distance:
                        closest_species = s
                        min_distance = distance

            if closest_species:
                closest_species.add(individual)
            else:
                # No species is similar enough, create a new species for this individual.
                self.__species.append(Species(individual))

        # python technical note:
        # we need a "working copy" list when removing elements while looping
        # otherwise we might end up having sync issues
        for s in self.__species[:]:
            # this happens when no genomes are compatible with the species
            if not s.members:
                #raise Exception('TODO: fix this')
                if report:
                    print("Removing species {0:d} for being empty".format(s.ID))
                # remove empty species
                self.__species.remove(s)

        self.__set_compatibility_threshold(report)

    def __set_compatibility_threshold(self, report):
        """ Controls compatibility threshold """
        t = self.config.compatibility_threshold
        dt = self.config.compatibility_change
        if len(self.__species) > self.config.species_size:
            t += dt
        elif len(self.__species) < self.config.species_size:
            t = max(0.0, t - dt)

        if self.config.compatibility_threshold != t:
            if report:
                print("Adjusted compatibility threshold to {0:f}".format(t))
            self.config.compatibility_threshold = t

    def __log_species(self):
        """ Logging species data for visualizing speciation and getting statistics"""
        temp_species_count = []
        temp_species_fitness = []
        if self.__species:
            higher = max([s.ID for s in self.__species])
            temp_species_fitness = range(1,higher+1)
            for i in range(1, higher + 1):
                found_species = False
                for s in self.__species:
                    temp_species_fitness[i-1] = "NA"
                    if i == s.ID:
                        temp_species_count.append(len(s.members))
                        temp_species_fitness[i-1] = s.get_average_fitness()
                        found_species = True
                        break
                if not found_species:
                    temp_species_count.append(0)
                    temp_species_fitness[i-1] = "NA"

        self.species_log.append(temp_species_count)
        self.species_fitness_log.append(temp_species_fitness)

    def epoch(self, fitness_function, n, report=True, save_best=False, checkpoint_interval=10,
              checkpoint_generation=None):
        """ Runs NEAT's genetic algorithm for n epochs.

            Keyword arguments:
            report -- show stats at each epoch (default True)
            save_best -- save the best genome from each epoch (default False)
            checkpoint_interval -- time in minutes between saving checkpoints (default 10 minutes)
            checkpoint_generation -- time in generations between saving checkpoints
                (default None -- option disabled)
        """
        t0 = time.time()  # for saving checkpoints

        for g in range(n):
            self.generation += 1

            if report:
                print('\n ****** Running generation {0:d} ****** \n'.format(self.generation))

            # Evaluate individuals
            fitness_function(self.population)
            # Speciates the population
            self.__speciate(report)

            # Current generation's best genome
            self.most_fit_genomes.append(copy.deepcopy(max(self.population)))
            # Current population's average fitness
            self.avg_fitness_scores.append(mean([c.fitness for c in self.population]))

            # Print some statistics
            best = self.most_fit_genomes[-1]

            # saves the best genome from the current generation
            if save_best:
                f = open('best_genome_' + str(self.generation), 'w')
                pickle.dump(best, f)
                f.close()

            # Stops the simulation
            if best.fitness > self.config.max_fitness_threshold:
                if report:
                    print('\nBest individual in epoch {0!s} meets fitness threshold - complexity: {1!s}'.format(
                        self.generation, best.size()))
                break

            # Remove stagnated species and its members (except if it has the best genome)
            for s in self.__species[:]:
                s.update_stagnation()
                if s.no_improvement_age > self.config.max_stagnation:
                    if report:
                        print("\n   Species {0:2d} (with {1:2d} individuals) is stagnated: removing it".format(s.ID, len(s.members)))
                    # removing species
                    self.__species.remove(s)
                    # removing all the species' members
                    # TODO: can be optimized!
                    for c in self.population[:]:
                        if c.species_id == s.ID:
                            self.population.remove(c)

            # Compute spawn levels for each remaining species
            self.diversity.compute_spawn_amount(self.__species)

            # Verify that all species received non-zero spawn counts, as the speciation mechanism
            # is intended to allow initially less-fit species time to improve before making them
            # extinct via the stagnation mechanism.
            for s in self.__species:
                assert s.spawn_amount > 0

            # Logging speciation stats
            self.__log_species()

            if report:
                if self.population:
                    std_dev = stdev([c.fitness for c in self.population])
                    print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(self.avg_fitness_scores[-1], std_dev))
                    print('Best fitness: {0:2.12} - size: {1!r} - species {2} - id {3}'.format(best.fitness, best.size(), best.species_id, best.ID))
                    print('Species length: {0:d} totaling {1:d} individuals'.format(len(self.__species), sum([len(s.members) for s in self.__species])))
                    print('Species ID       : {0!s}'.format([s.ID for s in self.__species]))
                    print('Each species size: {0!s}'.format([len(s.members) for s in self.__species]))
                    print('Amount to spawn  : {0!s}'.format([s.spawn_amount for s in self.__species]))
                    print('Species age      : {0}'.format([s.age for s in self.__species]))
                    print('Species avg fit  : {0!s}'.format([s.get_average_fitness() for s in self.__species]))
                    print('Species no improv: {0!s}'.format([s.no_improvement_age for s in self.__species]))
                else:
                    print('All species extinct.')

            # -------------------------- Producing new offspring -------------------------- #
            new_population = []  # next generation's population

            # If no species are left, create a new population from scratch, otherwise top off
            # population by reproducing existing species.
            if self.__species:
                for s in self.__species:
                    new_population.extend(s.reproduce(self.config))

                # Controls under or overflow  #
                fill = self.config.pop_size - len(new_population)
                if fill < 0:  # overflow
                    if report:
                        print('   Removing {0:d} excess individual(s) from the new population'.format(-fill))
                    # TODO: This is dangerous! I can't remove a species' representative!
                    new_population = new_population[:fill]  # Removing the last added members

                if fill > 0:  # underflow
                    if report:
                        print('   Producing {0:d} more individual(s) to fill up the new population'.format(fill))

                    while fill > 0:
                        # Selects a random genome from population
                        parent1 = random.choice(self.population)
                        # Search for a mate within the same species
                        found = False
                        for c in self.population:
                            # what if c is parent1 itself?
                            if c.species_id == parent1.species_id:
                                child = parent1.crossover(c)
                                new_population.append(child.mutate())
                                found = True
                                break
                        if not found:
                            # If no mate was found, just mutate it
                            new_population.append(parent1.mutate())
                        # new_population.append(genome.FFGenome.create_fully_connected())
                        fill -= 1

                assert self.config.pop_size == len(new_population), 'Different population sizes!'
                # Updates current population
                self.population = new_population
            else:
                self.__create_population()

            if checkpoint_interval is not None and time.time() > t0 + 60 * checkpoint_interval:
                self.__create_checkpoint(report)
                t0 = time.time()  # updates the counter
            elif checkpoint_generation is not None and self.generation % checkpoint_generation == 0:
                self.__create_checkpoint(report)
