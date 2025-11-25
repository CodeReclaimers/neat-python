"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""

import math
import random
from itertools import count

from neat.config import ConfigParameter, DefaultClassConfig
from neat.innovation import InnovationTracker
from neat.math_util import mean


# TODO: Provide some sort of optional cross-species performance criteria, which
# are then used to control stagnation and possibly the mutation rate
# configuration. This scheme should be adaptive so that species do not evolve
# to become "cautious" and only make very slow progress.


class DefaultReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 1)],
                                  'DefaultReproduction')

    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}
        
        # Create innovation tracker for tracking structural mutations
        # Per NEAT paper (Stanley & Miikkulainen, 2002), this persists across generations
        self.innovation_tracker = InnovationTracker()

    def create_new(self, genome_type, genome_config, num_genomes):
        """Create a new population of genomes from scratch."""
        # Set innovation tracker for initial genome creation
        genome_config.innovation_tracker = self.innovation_tracker
        
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
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

    def _adjust_spawn_exact(self, spawn_amounts, pop_size, min_species_size):
        """Adjust per-species spawn counts so that their sum matches pop_size exactly.

        This adjustment preserves the per-species minimum (min_species_size) and biases
        changes as follows:
        - When adding individuals (total < pop_size), smaller species are incremented first.
        - When removing individuals (total > pop_size), larger species are decremented first.
        """
        total_spawn = sum(spawn_amounts)
        if total_spawn == pop_size:
            return spawn_amounts

        num_species = len(spawn_amounts)
        min_total = num_species * min_species_size
        if min_total > pop_size:
            raise RuntimeError(
                "Configuration conflict: population size {} is less than "
                "num_species * min_species_size {} ({} * {}). Cannot satisfy per-species minima.".format(
                    pop_size, min_total, num_species, min_species_size
                )
            )

        diff = pop_size - total_spawn
        indexed = list(enumerate(spawn_amounts))

        if diff > 0:
            # Too few genomes overall: give extras to smaller species first.
            indexed.sort(key=lambda x: x[1])  # ascending by current spawn size
            i = 0
            while diff > 0 and indexed:
                idx, val = indexed[i]
                val += 1
                spawn_amounts[idx] = val
                indexed[i] = (idx, val)
                diff -= 1
                i = (i + 1) % len(indexed)
        else:
            # Too many genomes overall: remove from larger species first.
            remaining = -diff
            indexed.sort(key=lambda x: x[1], reverse=True)  # descending by current spawn size
            i = 0
            # We know a solution should exist whenever min_total <= pop_size, but
            # guard against pathological rounding behaviour with a safety break.
            while remaining > 0 and indexed:
                idx, val = indexed[i]
                if val > min_species_size:
                    val -= 1
                    spawn_amounts[idx] = val
                    indexed[i] = (idx, val)
                    remaining -= 1
                i = (i + 1) % len(indexed)
                if i == 0 and remaining > 0:
                    # Could not adjust further without violating the per-species minimum.
                    break

        if sum(spawn_amounts) != pop_size:
            raise RuntimeError(
                "Internal error adjusting spawn counts: could not match pop_size={} "
                "with min_species_size={}".format(pop_size, min_species_size)
            )

        return spawn_amounts

    def reproduce(self, config, species, pop_size, generation):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        
        Implements innovation tracking per NEAT paper (Stanley & Miikkulainen, 2002):
        The innovation tracker is reset at the start of each generation to enable
        same-generation deduplication while preserving the global innovation counter.
        """
        # Set innovation tracker for this generation and reset generation-specific tracking
        # This enables same-generation deduplication: if multiple genomes make the same
        # structural mutation this generation, they get the same innovation number
        config.genome_config.innovation_tracker = self.innovation_tracker
        self.innovation_tracker.reset_generation()
        
        # TODO: I don't like this modification of the species and stagnation objects,
        # because it requires internal knowledge of the objects.

        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {}  # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in afs.members.values()])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)  # type: float
        self.reporters.info(f"Average adjusted fitness: {avg_adjusted_fitness:.3f}")

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)
        # Adjust spawn counts so that the total exactly matches the requested
        # population size while respecting the per-species minimum.
        spawn_amounts = self._adjust_spawn_exact(spawn_amounts, pop_size, min_species_size)

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(s.members.items())
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness, with genome id as a
            # deterministic tie-breaker so that ordering (and thus parent
            # selection) is reproducible across runs and checkpoint restores.
            old_members.sort(reverse=True, key=lambda x: (x[1].fitness, x[0]))

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *
                                         len(old_members)))
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
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population
