import sys

from neat.species import Species
from neat.six_util import iteritems

# TODO: Add a method for the user to change the "is stagnant" computation.


class DefaultStagnation(object):
    def __init__(self, config, reporters):
        params = config.get_type_config(self)

        self.max_stagnation = int(params.get('max_stagnation'))
        self.species_fitness = params.get('species_fitness_func')

        if self.species_fitness == 'max':
            self.species_fitness_func = Species.max_fitness
        elif self.species_fitness == 'min':
            self.species_fitness_func = Species.min_fitness
        elif self.species_fitness == 'mean':
            self.species_fitness_func = Species.mean_fitness
        elif self.species_fitness == 'median':
            self.species_fitness_func = Species.median_fitness
        else:
            raise Exception("Unexpected species fitness: {0!r}".format(self.species_fitness))

        self.reporters = reporters

        self.previous_fitnesses = {}
        self.stagnant_counts = {}

    def remove(self, sid):
        self.previous_fitnesses.pop(sid, None)
        self.stagnant_counts.pop(sid, None)

    def update(self, species):
        result = []
        for sid, s in iteritems(species):
            fitness = self.species_fitness_func(s)
            prev_fitness = self.previous_fitnesses.get(sid, -sys.float_info.max)
            if fitness > prev_fitness:
                scount = 0
            else:
                scount = self.stagnant_counts.get(sid, 0) + 1

            self.previous_fitnesses[sid] = fitness
            self.stagnant_counts[sid] = scount

            is_stagnant = scount >= self.max_stagnation
            result.append((sid, s, is_stagnant))

            if is_stagnant:
                self.remove(s)

        self.reporters.info('Species no improv: {0!r}'.format(self.stagnant_counts))

        return result