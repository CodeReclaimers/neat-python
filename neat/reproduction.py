import math
import random


# TODO: Provide some sort of optional cross-species performance criteria, which
# are then used to control stagnation and possibly the mutation rate configuration.
# This scheme should be adaptive so that species do not evolve to become "cautious"
# and only make very slow progress.


class DefaultReproduction(object):
    """
    Implements the default NEAT-python reproduction scheme: explicit fitness sharing with fixed-time
    species stagnation.
    """
    def __init__(self, config, reporters, genome_indexer, innovation_indexer):
        self.config = config
        self.reporters = reporters
        self.genome_indexer = genome_indexer
        self.innovation_indexer = innovation_indexer
        self.stagnation = config.stagnation_type(self.config)
        #self.diversity = config.diversity_type(self.config)

    def reproduce(self, species):
        # Filter out stagnated species and collect the set of non-stagnated species members.
        remaining_species = {}
        remaining_population = []
        for s, stagnant in self.stagnation.update(species):
            if stagnant:
                self.reporters.species_stagnant(s)
            else:
                remaining_species[s.ID] = s

                # Compute adjusted fitness.
                for m in s.members:
                    remaining_population.append((m.fitness / len(s.members), m))

        new_population = []
        new_species = []
        if remaining_population:
            # Sort in order of descending adjusted fitness.
            remaining_population.sort(reverse=True)
            #print remaining_population
            #print [(af, m.fitness) for af, m in remaining_population]

            # Determine the cutoff adjusted fitness for reproduction.
            repro_cutoff = int(math.ceil(self.config.survival_threshold * self.config.pop_size))
            remaining_population = remaining_population[:repro_cutoff]
            cutoff_fitness = remaining_population[-1][0]
            #print remaining_population
            #print [(af, m.fitness) for af, m in remaining_population]
            #print cutoff_fitness

            # Update species membership to remove any unfit members, and transfer elites (if any)
            # to the new population.
            for s in remaining_species.values():
                n = len(s.members)
                s.members = [m for m in s.members if (m.fitness / n) >= cutoff_fitness]
                if s.members:
                    s.representative = random.choice(s.members)

                    if self.config.elitism > 0:
                        s.members.sort(key=lambda m: m.fitness, reverse=True)
                        new_population.extend(s.members[:self.config.elitism])

                    new_species.append(s)

            # Randomly choose parents and produce offspring until the population is restored.
            while len(new_population) < self.config.pop_size:
                parent1_af, parent1 = random.choice(remaining_population)
                parent_species = remaining_species[parent1.species_id]
                parent2 = random.choice(parent_species.members)

                # Note that if the parents are not distinct, crossover should produce a
                # genetically identical clone of the parent (but with a different ID).
                child = parent1.crossover(parent2, self.genome_indexer.next())
                new_population.append(child.mutate(self.innovation_indexer))

        new_species.sort(key=lambda s: s.ID)
        for s in new_species:
            s.members = []

        return new_species, new_population