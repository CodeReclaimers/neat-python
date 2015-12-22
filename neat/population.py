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
from neat.diversity import ExplicitFitnessSharing

class MassExtinctionException(Exception):
    pass


class Population(object):
    """ Manages all the species  """

    def __init__(self, config, checkpoint_file=None, initial_population=None,
                 node_gene_type=NodeGene, conn_gene_type=ConnectionGene,
                 diversity_type=ExplicitFitnessSharing):

        # If config is not a Config object, assume it is a path to the config file.
        if not isinstance(config, Config):
            config = Config(config)

        self.config = config
        # TODO: Move node_gene_type, conn_gene_type, and diversity_type to the configuration object.
        self.node_gene_type = node_gene_type
        self.conn_gene_type = conn_gene_type
        self.diversity = diversity_type(self.config)

        self.population = None
        self.species = []
        self.species_log = []
        self.fitness_scores = []
        self.most_fit_genomes = []
        self.generation = -1
        self.total_evaluations = 0

        if checkpoint_file:
            assert initial_population is None
            self._load_checkpoint(checkpoint_file)
        elif initial_population is None:
            self._create_population()
        else:
            self.population = initial_population

        # Partition the population into species based on current configuration.
        self._speciate()

    def _load_checkpoint(self, checkpoint):
        '''Resumes the simulation from a previous saved point.'''
        with gzip.open(checkpoint) as f:
            print('Resuming from a previous point: {0}'.format(checkpoint))

            (self.population,
             self.species,
             self.species_log,
             self.fitness_scores,
             self.most_fit_genomes,
             self.generation,
             random_state) = pickle.load(f)

            random.setstate(random_state)

    def _create_checkpoint(self):
        """ Save the current simulation state. """
        fn = 'neat-checkpoint-{0}'.format(self.generation)
        with gzip.open(fn, 'w', compresslevel=5) as f:
            data = (self.population,
                    self.species,
                    self.species_log,
                    self.fitness_scores,
                    self.most_fit_genomes,
                    self.generation,
                    random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _create_population(self):
        if self.config.feedforward:
            genotype = FFGenome
        else:
            genotype = Genome

        self.population = []
        # TODO: Add FS-NEAT support, which creates an empty connection set, and then performs a
        # single add connection mutation.
        # This would give three initialization methods:
        # 1. Fully connected (each input connected to all outputs)
        # 2. "minimally" connected, which isn't really minimal (one random input to each output)
        # 3. FS-NEAT connected (one random connection)
        if self.config.fully_connected:
            for i in range(self.config.pop_size):
                g = genotype.create_fully_connected(self.config, self.node_gene_type, self.conn_gene_type)
                self.population.append(g)
        else:
            for i in range(self.config.pop_size):
                g = genotype.create_minimally_connected(self.config, self.node_gene_type, self.conn_gene_type)
                self.population.append(g)

        if self.config.hidden_nodes > 0:
            for g in self.population:
                g.add_hidden_nodes(self.config.hidden_nodes)

    def _speciate(self):
        """Group genomes into species by genetic similarity."""
        for individual in self.population:
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
            if not s.members:
                raise Exception('TODO: fix empty species bug')

    def _log_stats(self):
        """ Gather data for visualization/reporting purposes. """
        species_sizes = dict((s.ID, len(s.members)) for s in self.species)
        self.species_log.append(species_sizes)
        self.most_fit_genomes.append(copy.deepcopy(max(self.population)))
        self.fitness_scores.append([c.fitness for c in self.population])

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
                print('\n ****** Running generation {0} ****** \n'.format(self.generation))

            # Evaluate individuals
            fitness_function(self.population)
            self.total_evaluations += len(self.population)

            # Gather statistics.
            self._log_stats()

            # Print some statistics
            best = self.most_fit_genomes[-1]
            if report:
                fit_mean = mean([c.fitness for c in self.population])
                fit_std = stdev([c.fitness for c in self.population])
                print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
                print('Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best.fitness, best.size(),
                                                                                           best.species_id, best.ID))
                print('Species length: {0:d} totaling {1:d} individuals'.format(len(self.species), sum([len(s.members) for s in self.species])))
                print('Species ID       : {0!s}'.format([s.ID for s in self.species]))
                print('Each species size: {0!s}'.format([len(s.members) for s in self.species]))
                print('Amount to spawn  : {0!s}'.format([s.spawn_amount for s in self.species]))
                print('Species age      : {0}'.format([s.age for s in self.species]))
                print('Species avg fit  : {0!r}'.format([s.get_average_fitness() for s in self.species]))
                print('Species no improv: {0!r}'.format([s.no_improvement_age for s in self.species]))

            # Saves the best genome from the current generation if requested.
            if save_best:
                f = open('best_genome_' + str(self.generation), 'w')
                pickle.dump(best, f)
                f.close()

            # End when the fitness threshold is reached.
            if best.fitness >= self.config.max_fitness_threshold:
                if report:
                    print('\nBest individual in epoch {0} meets fitness threshold - complexity: {1!r}'.format(
                        self.generation, best.size()))
                break

            # Remove stagnated species.
            #TODO: Log species removal for visualization purposes.
            new_species = []
            for s in self.species:
                s.update_stagnation()
                if s.no_improvement_age <= self.config.max_stagnation:
                    new_species.append(s)
                else:
                    if report:
                        print("\n   Species {0} with {1} members is stagnated: removing it".format(s.ID, len(s.members)))
            self.species = new_species

            # Check for complete extinction.
            if not self.species:
                if report:
                    print('All species extinct.')
                raise MassExtinctionException()

            # Compute spawn levels for all current species and then reproduce.
            self.diversity.compute_spawn_amount(self.species)
            self.population = []
            for s in self.species:
                # Verify that all species received non-zero spawn counts, as the speciation mechanism
                # is intended to allow initially less fit species time to improve before making them
                # extinct via the stagnation mechanism.
                assert s.spawn_amount > 0
                self.population.extend(s.reproduce(self.config))

            self._speciate()

            if checkpoint_interval is not None and time.time() > t0 + 60 * checkpoint_interval:
                if report:
                    print('Creating timed checkpoint file at generation: {0}'.format(self.generation))
                self._create_checkpoint()

                # Update the checkpoint time.
                t0 = time.time()
            elif checkpoint_generation is not None and self.generation % checkpoint_generation == 0:
                if report:
                    print('Creating generation checkpoint file at generation: {0}'.format(self.generation))
                self._create_checkpoint()
