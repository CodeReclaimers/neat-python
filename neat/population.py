from __future__ import print_function

import gzip
import pickle
import random
import time

from neat.config import Config
from neat.indexer import Indexer
from neat.reporting import ReporterSet, StatisticsReporter, StdOutReporter
from neat.species import Species


class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core NEAT algorithm.  It maintains a list of Species instances,
    each of which contains a collection of Genome instances.
    """

    def __init__(self, config, initial_population=None):
        """
        :param config: Either a config.Config object or path to a configuration file.
        :param initial_population: Either an initial set of Genome instances to be used
               as the initial population, or None, in which case a randomized set of Genomes
               will be created automatically based on the configuration parameters.
        """

        # If config is not a Config object, assume it is a path to the config file.
        if not isinstance(config, Config):
            config = Config(config)

        # Configure statistics and reporting as requested by the user.
        self.reporters = ReporterSet()
        if config.collect_statistics:
            self.statistics = StatisticsReporter()
            self.add_reporter(self.statistics)
        else:
            self.statistics = None

        if config.report:
            self.add_reporter(StdOutReporter())

        self.config = config
        self.species_indexer = Indexer(1)
        self.genome_indexer = Indexer(1)
        self.reproduction = config.reproduction_type(self.config, self.reporters,
                                                     self.genome_indexer)

        self.species = []
        self.generation = -1
        self.total_evaluations = 0

        # Create a population if one is not given, then partition into species.
        if initial_population is None:
            initial_population = self._create_population()
        self._speciate(initial_population)

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def load_checkpoint(self, filename):
        '''Resumes the simulation from a previous saved point.'''
        self.reporters.loading_checkpoint(filename)
        with gzip.open(filename) as f:
            (self.species,
             self.generation,
             random_state) = pickle.load(f)

            random.setstate(random_state)

    def save_checkpoint(self, filename=None, checkpoint_type="user"):
        """ Save the current simulation state. """
        if filename is None:
            filename = 'neat-checkpoint-{0}'.format(self.generation)

        self.reporters.saving_checkpoint(checkpoint_type, filename)

        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (self.species,
                    self.generation,
                    random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _create_population(self):
        # Create a collection of unconnected genomes with no hidden nodes.
        new_population = []

        # TODO: The genotype class should know how to do everything below, based
        # solely on what's in the config object. This allows users to completely
        # replace the initial population creation scheme if they choose.
        for i in range(self.config.pop_size):
            g_id = self.genome_indexer.get_next()
            g = self.config.genotype.create_unconnected(g_id, self.config)
            new_population.append(g)

        # Add hidden nodes if requested.
        if self.config.hidden_nodes > 0:
            for g in new_population:
                g.add_hidden_nodes(self.config.hidden_nodes)

        # Add connections based on initial connectivity type.
        if self.config.initial_connection == 'fs_neat':
            for g in new_population:
                g.connect_fs_neat()
        elif self.config.initial_connection == 'fully_connected':
            for g in new_population:
                g.connect_full()
        elif self.config.initial_connection == 'partial':
            for g in new_population:
                g.connect_partial(self.config.connection_fraction)

        return new_population

    def _speciate(self, population):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        for individual in population:
            # Find the species with the most similar representative.
            min_distance = None
            closest_species = None
            for s in self.species:
                distance = individual.distance(s.representative)
                if distance < self.config.compatibility_threshold \
                    and (min_distance is None or distance < min_distance):
                        closest_species = s
                        min_distance = distance

            if closest_species:
                closest_species.add(individual)
            else:
                # No species is similar enough, create a new species for this individual.
                self.species.append(Species(individual, self.species_indexer.get_next()))

        # Only keep non-empty species.
        self.species = [s for s in self.species if s.members]

        # Select a random current member as the new representative.
        for s in self.species:
            s.representative = random.choice(s.members)

    def run(self, fitness_function, n):
        """
        Runs NEAT's genetic algorithm for n generations.

        The user-provided fitness_function should take one argument, a list of all genomes in the population,
        and its return value is ignored.  This function is free to maintain external state, perform evaluations
        in parallel, and probably any other thing you want.  The only requirement is that each individual's
        fitness member must be set to a floating point value after this function returns.

        It is assumed that fitness_function does not modify the list of genomes, or the genomes themselves, apart
        from updating the fitness member.
        """

        # Remember start time for saving timed checkpoints.
        last_checkpoint = time.time()

        for g in range(n):
            self.generation += 1

            self.reporters.start_generation(self.generation)

            # Collect a list of all members from all species.
            population = []
            for s in self.species:
                population.extend(s.members)

            # Evaluate all individuals in the population using the user-provided function.
            # TODO: Add an option to only evaluate each genome once, to reduce number of
            # fitness evaluations in cases where the fitness is known to be the same if the
            # genome doesn't change--in these cases, evaluating unmodified elites in each
            # generation is a waste of time.
            fitness_function(population)
            self.total_evaluations += len(population)

            # Gather and report statistics.
            best = max(population)
            self.reporters.post_evaluate(population, self.species, best)

            # Save the best genome from the current generation if requested.
            if self.config.save_best:
                with open('best_genome_' + str(self.generation), 'wb') as f:
                    pickle.dump(best, f)

            # End if the fitness threshold is reached.
            if best.fitness >= self.config.max_fitness_threshold:
                self.reporters.found_solution(self.generation, best)
                break

            # Create the next generation from the current generation.
            self.species, new_population = self.reproduction.reproduce(self.species, self.config.pop_size)

            # Check for complete extinction
            if not self.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    new_population = self._create_population()
                else:
                    raise CompleteExtinctionException()

            # Update species age.
            for s in self.species:
                s.age += 1

            # Divide the new population into species.
            self._speciate(new_population)

            # Save checkpoints if necessary.
            if self.config.checkpoint_time_interval is not None:
                timed_checkpoint_due = last_checkpoint + 60 * self.config.checkpoint_time_interval
                if time.time() >= timed_checkpoint_due:
                    self.save_checkpoint(checkpoint_type="timed")
                    last_checkpoint = time.time()

            if self.config.checkpoint_gen_interval is not None \
                    and self.generation % self.config.checkpoint_gen_interval == 0:
                self.save_checkpoint(checkpoint_type="generation")

            self.reporters.end_generation()
