import math
import random


# TODO: Provide some sort of optional cross-species performance criteria, which
# are then used to control stagnation and possibly the mutation rate configuration.
# This scheme should be adaptive so that species do not evolve to become "cautious"
# and only make very slow progress.


class DefaultReproduction(object):
    """
    Implements the default NEAT-python reproduction scheme: explicit fitness sharing
    with fixed-time species stagnation.
    """
    def __init__(self, config, reporters, genome_indexer):
        params = config.get_type_config(self)
        self.elitism = int(params.get('elitism'))
        self.survival_threshold = float(params.get('survival_threshold'))

        self.reporters = reporters
        self.genome_indexer = genome_indexer
        self.stagnation = config.stagnation_type(config, reporters)

    def reproduce(self, species, pop_size):
        # Filter out stagnated species and collect the set of non-stagnated species members.
        remaining_species = {}
        species_fitness = []
        avg_adjusted_fitness = 0.0
        for s, stagnant in self.stagnation.update(species):
            if stagnant:
                self.reporters.species_stagnant(s)
            else:
                remaining_species[s.ID] = s

                # Compute adjusted fitness.
                species_sum = 0.0
                for m in s.members:
                    af = m.fitness / len(s.members)
                    species_sum += af

                sfitness = species_sum / len(s.members)
                species_fitness.append((s, sfitness))
                avg_adjusted_fitness += sfitness

        # No species left.
        if not remaining_species:
            return [], []

        avg_adjusted_fitness /= len(species_fitness)
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new individuals to create for the new generation.
        spawn_amounts = []
        for s, sfitness in species_fitness:
            spawn = len(s.members)
            if sfitness > avg_adjusted_fitness:
                spawn *= 1.1
            else:
                spawn *= 0.9
            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [int(round(n * norm)) for n in spawn_amounts]
        self.reporters.info("Spawn amounts: {0}".format(spawn_amounts))
        self.reporters.info('Species fitness  : {0!r}'.format([sfitness for s, sfitness in species_fitness]))

        new_population = []
        new_species = []
        for spawn, (s, sfitness) in zip(spawn_amounts, species_fitness):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.elitism)

            if spawn <= 0:
                continue

            # The species has at least one member for the next generation, so retain it.
            old_members = s.members
            s.members = []
            new_species.append(s)

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True)

            # Transfer elites to new generation.
            if self.elitism > 0:
                new_population.extend(old_members[:self.elitism])
                spawn -= self.elitism

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

                parent1 = random.choice(old_members)
                parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                child = parent1.crossover(parent2, self.genome_indexer.get_next())
                new_population.append(child.mutate())

        # Sort species by ID (purely for ease of reading the reported list).
        new_species.sort(key=lambda sp: sp.ID)

        return new_species, new_population