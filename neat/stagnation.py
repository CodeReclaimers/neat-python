import sys


# TODO: Add a method for the user to change the "is stagnant" computation.

class FixedStagnation(object):
    def __init__(self, config, reporters):
        self.config = config
        self.reporters = reporters

        self.previous_fitnesses = {}
        self.stagnant_counts = {}

    def remove(self, species):
        if species.ID in self.previous_fitnesses:
            del self.previous_fitnesses[species.ID]

        if species.ID in self.stagnant_counts:
            del self.stagnant_counts[species.ID]

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

            is_stagnant = scount >= self.config.max_stagnation
            result.append((s, is_stagnant))

            if is_stagnant:
                self.remove(s)

        self.reporters.info('Species no improv: {0!r}'.format(self.stagnant_counts))

        return result