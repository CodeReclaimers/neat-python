import sys
from neat.math_util import mean


def species_max_fitness(species):
    return max([m.fitness for m in species.members])


def species_min_fitness(species):
    return min([m.fitness for m in species.members])


def species_mean_fitness(species):
    return mean([m.fitness for m in species.members])


def species_median_fitness(species):
    fitnesses = [m.fitness for m in species.members]
    fitnesses.sort()
    return fitnesses[len(fitnesses) // 2]


# TODO: Add a method for the user to change the "is stagnant" computation.

class DefaultStagnation(object):
    def __init__(self, config, reporters):
        params = config.get_type_config(self)

        self.max_stagnation = int(params.get('max_stagnation'))
        self.species_fitness = params.get('species_fitness_func')

        if self.species_fitness == 'max':
            self.species_fitness_func = species_max_fitness
        elif self.species_fitness == 'min':
            self.species_fitness_func = species_min_fitness
        elif self.species_fitness == 'mean':
            self.species_fitness_func = species_mean_fitness
        elif self.species_fitness == 'median':
            self.species_fitness_func = species_median_fitness
        else:
            raise Exception("Unexpected species fitness: {0!r}".format(self.species_fitness))

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
            fitness = self.species_fitness_func(s)
            scount = self.stagnant_counts.get(s.ID, 0)
            prev_fitness = self.previous_fitnesses.get(s.ID, -sys.float_info.max)
            if fitness > prev_fitness:
                scount = 0
            else:
                scount += 1

            self.previous_fitnesses[s.ID] = fitness
            self.stagnant_counts[s.ID] = scount

            is_stagnant = scount >= self.max_stagnation
            result.append((s, is_stagnant))

            if is_stagnant:
                self.remove(s)

        self.reporters.info('Species no improv: {0!r}'.format(self.stagnant_counts))

        return result