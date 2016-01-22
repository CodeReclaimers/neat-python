import sys


# TODO: Add a method for the user to change the "is stagnant" computation.

class FixedStagnation(object):
    def __init__(self, config):
        self.config = config

        self.previous_fitnesses = {}
        self.stagnant_counts = {}

    def update(self, species):
        result = []
        for s in species:
            fitness = self.config.species_fitness_func(s)
            scount = self.stagnant_counts.get(s.ID, 0)
            prev_fitness = self.previous_fitnesses.get(s.ID, -sys.float_info.max)
            if fitness > prev_fitness:
                scount = 0
            else:
                scount += 1

            self.previous_fitnesses[s.ID] = fitness
            self.stagnant_counts[s.ID] = scount

            result.append((s, scount >= self.config.max_stagnation))

        return result