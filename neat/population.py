from __future__ import print_function

import copy
import gzip
import pickle
import random
import time

from neat.config import Config
from neat.diversity import ExplicitFitnessSharing
from neat.genome import Genome, FFGenome
from neat.math_util import mean, stdev
from neat.species import Species


class MassExtinctionException(Exception):
    pass


class Population(object):
    """ Manages all the species  """

    def __init__(self, config, initial_population=None,
                 diversity_type=ExplicitFitnessSharing, report=True, save_best=False,
                 checkpoint_interval=10, checkpoint_generation=None):
        """
        :param config: Either a config.Config object or path to a configuration file.
        :param initial_population:
        :param diversity_type:
        :param report: Show stats after each generation.
        :param save_best: Save the best genome from each generation.
        :param checkpoint_interval: Time in minutes between saving checkpoints.
        :param checkpoint_generation: Time in generations between saving checkpoints.
        """

        # If config is not a Config object, assume it is a path to the config file.
        if not isinstance(config, Config):
            config = Config(config)

        self.config = config
        # TODO: Move diversity_type to the configuration object.
        self.diversity = diversity_type(self.config)

        self.species = []
        self.generation_statistics = []
        self.most_fit_genomes = []
        self.generation = -1
        self.total_evaluations = 0

        if initial_population is None:
            initial_population = self._create_population()

        # Partition the population into species based on current configuration.
        self._speciate(initial_population)

        self.report = report
        self.save_best = save_best
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_generation = checkpoint_generation

    def __del__(self):
        Species.clear_indexer()

    def load_checkpoint(self, filename):
        '''Resumes the simulation from a previous saved point.'''
        with gzip.open(filename) as f:
            print('Resuming from a previous point: ' + filename)

            (self.species,
             self.generation_statistics,
             self.most_fit_genomes,
             self.generation,
             random_state) = pickle.load(f)

            random.setstate(random_state)

    def save_checkpoint(self, filename=None):
        """ Save the current simulation state. """
        if filename is None:
            filename = 'neat-checkpoint-{0}'.format(self.generation)
        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (self.species,
                    self.generation_statistics,
                    self.most_fit_genomes,
                    self.generation,
                    random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _create_population(self):

        # Create a collection of unconnected genomes with no hidden nodes.
        new_population = []
        for i in range(self.config.pop_size):
            new_population.append(self.config.genotype.create_unconnected(self.config))

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

        return new_population

    def _speciate(self, population):
        """Place genomes into species by genetic similarity."""
        for individual in population:
            # Find the species with the most similar representative.
            min_distance = None
            closest_species = None
            for s in self.species:
                distance = individual.distance(s.representative)
                if distance < self.config.compatibility_threshold:
                    if min_distance is None or distance < min_distance:
                        closest_species = s
                        min_distance = distance

            if closest_species:
                closest_species.add(individual)
            else:
                # No species is similar enough, create a new species for this individual.
                self.species.append(Species(individual))

        # Verify that no species are empty.
        for s in self.species:
            assert s.members

    def _log_stats(self, population):
        """ Gather data for visualization/reporting purposes. """
        # Keep a deep copy of the best genome, so that future modifications to the genome
        # do not produce an unexpected change in statistics.
        self.most_fit_genomes.append(copy.deepcopy(max(population)))

        # Store the fitnesses of the members of each currently active species.
        species_stats = {}
        for s in self.species:
            species_stats[s.ID] = [c.fitness for c in s.members]
        self.generation_statistics.append(species_stats)

    def epoch(self, fitness_function, n):
        """ Runs NEAT's genetic algorithm for n epochs. """
        t0 = time.time()  # for saving checkpoints

        for g in range(n):
            self.generation += 1

            if self.report:
                print('\n ****** Running generation {0} ****** \n'.format(self.generation))

            gen_start = time.time()

            # Collect a list of all members from all species.
            population = []
            for s in self.species:
                population.extend(s.members)

            # Evaluate individuals
            fitness_function(population)
            self.total_evaluations += len(population)

            # Gather statistics.
            self._log_stats(population)

            # Print some statistics
            best = self.most_fit_genomes[-1]
            if self.report:
                fit_mean = mean([c.fitness for c in population])
                fit_std = stdev([c.fitness for c in population])
                print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
                print('Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best.fitness, best.size(),
                                                                                           best.species_id, best.ID))
                print('Species length: {0:d} totaling {1:d} individuals'.format(len(self.species), sum(
                    [len(s.members) for s in self.species])))
                print('Species ID       : {0!s}'.format([s.ID for s in self.species]))
                print('Each species size: {0!s}'.format([len(s.members) for s in self.species]))
                print('Amount to spawn  : {0!s}'.format([s.spawn_amount for s in self.species]))
                print('Species age      : {0}'.format([s.age for s in self.species]))
                print('Species avg fit  : {0!r}'.format([s.get_average_fitness() for s in self.species]))
                print('Species no improv: {0!r}'.format([s.no_improvement_age for s in self.species]))

            # Saves the best genome from the current generation if requested.
            if self.save_best:
                with open('best_genome_' + str(self.generation), 'w') as f:
                    pickle.dump(best, f)

            # End when the fitness threshold is reached.
            if best.fitness >= self.config.max_fitness_threshold:
                if self.report:
                    print('\nBest individual in epoch {0} meets fitness threshold - complexity: {1!r}'.format(
                        self.generation, best.size()))
                break

            # Remove stagnated species.
            # TODO: Log species removal for visualization purposes.
            new_species = []
            for s in self.species:
                s.update_stagnation()
                if s.no_improvement_age <= self.config.max_stagnation:
                    new_species.append(s)
                else:
                    if self.report:
                        print("\n   Species {0} with {1} members is stagnated: removing it".format(s.ID, len(s.members)))
            self.species = new_species

            # Check for complete extinction.
            new_population = []
            if not self.species:
                if self.report:
                    print('All species extinct.')

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    new_population = self._create_population()
                else:
                    raise MassExtinctionException()
            else:
                # Compute spawn levels for all current species and then reproduce.
                self.diversity.compute_spawn_amount(self.species)
                for s in self.species:
                    # Verify that all species received non-zero spawn counts, as the speciation mechanism
                    # is intended to allow initially less fit species time to improve before making them
                    # extinct via the stagnation mechanism.
                    assert s.spawn_amount > 0
                    # The Species.reproduce keeps one random child as its new representative, and
                    # returns the rest as a list, which must be sorted into species.
                    new_population.extend(s.reproduce(self.config))

            self._speciate(new_population)

            if self.checkpoint_interval is not None and time.time() > t0 + 60 * self.checkpoint_interval:
                if self.report:
                    print('Creating timed checkpoint file at generation: {0}'.format(self.generation))
                self.save_checkpoint()

                # Update the checkpoint time.
                t0 = time.time()
            elif self.checkpoint_generation is not None and self.generation % self.checkpoint_generation == 0:
                if self.report:
                    print('Creating generation checkpoint file at generation: {0}'.format(self.generation))
                self.save_checkpoint()

            if self.report:
                print("Generation time: {0:.3f} sec".format(time.time() - gen_start))