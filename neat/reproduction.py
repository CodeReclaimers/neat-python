from __future__ import division

import math
import random

from neat.indexer import Indexer
from neat.math_util import mean
from neat.six_util import iteritems, itervalues

# TODO: Provide some sort of optional cross-species performance criteria, which
# are then used to control stagnation and possibly the mutation rate configuration.
# This scheme should be adaptive so that species do not evolve to become "cautious"
# and only make very slow progress.


class DefaultReproduction(object):
    """
    Handles creation of genomes, either from scratch or by sexual or asexual
    reproduction from parents. Implements the default NEAT-python reproduction
    scheme: explicit fitness sharing with fixed-time species stagnation.
    """

    # TODO: Create a separate configuration class instead of using a dict (for consistency with other types).
    @classmethod
    def parse_config(cls, param_dict):
        config = {'elitism': 1,
                  'survival_threshold': 0.2,
                  'min_species_size': 2}
        config.update(param_dict)

        return config

    @classmethod
    def write_config(cls, f, param_dict):
        elitism = param_dict.get('elitism', 1)
        f.write('elitism            = {}\n'.format(elitism))
        survival_threshold = param_dict.get('survival_threshold', 0.2)
        f.write('survival_threshold = {}\n'.format(survival_threshold))

    def __init__(self, config, reporters, stagnation):
        self.elitism = int(config.get('elitism'))
        self.survival_threshold = float(config.get('survival_threshold'))

        self.reporters = reporters
        self.genome_indexer = Indexer(1)
        self.stagnation = stagnation
        self.ancestors = {}

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = self.genome_indexer.get_next()
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        return spawn_amounts

    def reproduce(self, config, species, pop_size, generation):
        # TODO: I don't like this modification of the species and stagnation objects,
        # because it requires internal knowledge of the objects.

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        all_fitnesses = []
        for sid, s in iteritems(species.species):
            all_fitnesses.extend(m.fitness for m in itervalues(s.members))
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        fitness_range = max(1.0, max_fitness - min_fitness)

        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        remaining_species = []
        for sid, s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(sid, s)
            else:
                # Compute adjusted fitness.
                msf = mean([m.fitness for m in itervalues(s.members)])
                af = (msf - min_fitness) / fitness_range
                s.adjusted_fitness = af
                remaining_species.append(s)

        # No species left.
        if not remaining_species:
            species.species = {}
            return []

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new memebers for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = float(config.reproduction_config['min_species_size'])
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(iteritems(s.members))
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.elitism > 0:
                for i, m in old_members[:self.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.survival_threshold * len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = self.genome_indexer.get_next()
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population