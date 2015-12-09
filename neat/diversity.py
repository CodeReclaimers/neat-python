'''
Implementations of diversity maintenance schemes.
'''


class AgedFitnessSharing(object):
    '''
    This class encapsulates a fitness sharing scheme. It is responsible for
    computing the number of individuals to be spawned for each species in the
    next generation, based on species fitness, age, and size.

    Fitness inside a species is shared by all its members, so that a species
    that happens to end up with a large initial number of members is less
    likely to dominate the entire population.
    '''

    def __init__(self, config):
        self.config = config

    def compute_spawn_amount(self, species):
        if not species:
            return

        # Get average fitnesses and their range.
        fitnesses = [s.average_fitness() for s in species]
        min_fitness = min(fitnesses)
        max_fitness = max(fitnesses)
        # This shift is used to compute an adjusted fitness that is
        # positive-valued, with a magnitude difference of ~2 between
        # most fit and least fit.
        fitness_shift = max_fitness - 2 * min_fitness + 1

        adjusted_fitnesses = []
        total_adjusted_fitness = 0.0
        for f, s in zip(fitnesses, species):
            # Make all adjusted fitnesses positive, and apply adjustment for population size.
            af = (f + fitness_shift) / len(s.members)

            # Apply adjustments for species age.
            if s.age < self.config.youth_threshold:
                af *= self.config.youth_boost
            elif s.age > self.config.old_threshold:
                af *= self.config.old_penalty

            adjusted_fitnesses.append(af)
            total_adjusted_fitness += af

        # Distribute spawn amounts among the species based on their share of adjusted fitness.
        r = self.config.pop_size / total_adjusted_fitness
        for s, af in zip(species, adjusted_fitnesses):
            s.spawn_amount = int(round((af * r)))
