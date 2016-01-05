'''
Implementations of diversity maintenance schemes.
'''
from math import ceil


class ExplicitFitnessSharing(object):
    '''
    This class encapsulates a fitness sharing scheme. It is responsible for
    computing the number of individuals to be spawned for each species in the
    next generation, based on species fitness and size.

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
        # TODO: Separate the species fitness computation and the spawn allotments into
        # different mechanisms, to allow more easily changing/testing schemes.
        fitnesses = [s.get_average_fitness() for s in species]
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
            adjusted_fitnesses.append(af)
            total_adjusted_fitness += af

        # Distribute spawn amounts among the species based on their share of adjusted fitness.
        # Each species is guaranteed a minimum spawn of survival_threshold * (current member count).
        r = self.config.pop_size * (1 - self.config.survival_threshold) / total_adjusted_fitness
        for s, af in zip(species, adjusted_fitnesses):
            min_spawn = len(s.members) * self.config.survival_threshold
            s.spawn_amount = int(ceil((min_spawn + af * r)))
            assert s.spawn_amount > 0
