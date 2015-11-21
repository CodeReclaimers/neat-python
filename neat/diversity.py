'''
Implementations of diversity maintenance rules.
'''

class AgedFitnessSharing(object):
    '''
    This class encapsulates a fitness sharing scheme. It is responsible for computing the number of
    individuals to be spawned for each species in the next generation, based on species fitness, age, and size.
    '''
    def __init__(self, config):
        self.config = config

    def compute_spawn_amount(self, species):
        # Get each species' average fitness, and adjust for species age and size.
        adjusted_fitnesses = []
        total_adjusted_fitness = 0.0
        for s in species:
            # Fitness inside a species is shared by all its members, so that one species
            # is less likely to dominate the entire population.
            af = s.average_fitness() / len(s.members)

            # Apply adjustments for species age.
            if s.age < self.config.youth_threshold:
                af *= self.config.youth_boost
            elif s.age > self.config.old_threshold:
                af *= self.config.old_penalty

            adjusted_fitnesses.append(af)
            total_adjusted_fitness += af

        # Compute target size for next generation.
        r = self.config.pop_size / total_adjusted_fitness
        for s, af in zip(species, adjusted_fitnesses):
            s.spawn_amount = int(round((af * r)))
