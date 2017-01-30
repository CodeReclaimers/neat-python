import sys

from neat.six_util import iteritems
from neat.math_util import stat_functions

# TODO: Add a method for the user to change the "is stagnant" computation.


class DefaultStagnation(object):
    @classmethod
    def parse_config(cls, param_dict):
        config = {'species_fitness_func': 'mean',
                  'max_stagnation': 15,
                  'species_elitism': 0}
        config.update(param_dict)

        return config

    @classmethod
    def write_config(cls, f, config):
        fitness_func = config.get('species_fitness_func', 'mean')
        f.write('species_fitness_func = {}\n'.format(fitness_func))
        max_stagnation = config.get('max_stagnation', 15)
        f.write('max_stagnation       = {}\n'.format(max_stagnation))
        species_elitism = config.get('species_elitism', 15)
        f.write('species_elitism      = {}\n'.format(species_elitism))

    def __init__(self, config, reporters):
        self.max_stagnation = int(config.get('max_stagnation'))
        self.species_fitness = config.get('species_fitness_func')
        self.species_elitism = int(config.get('species_elitism'))

        self.species_fitness_func = stat_functions.get(self.species_fitness)
        if self.species_fitness_func is None:
            raise Exception("Unexpected species fitness: {0!r}".format(self.species_fitness))

        self.reporters = reporters

    def update(self, species_set, generation):
        species_data = []
        for sid, s in iteritems(species_set.species):
            if s.fitness_history:
                prev_fitness = max(s.fitness_history)
            else:
                prev_fitness = -sys.float_info.max

            s.fitness = self.species_fitness_func(s.get_fitnesses())
            s.fitness_history.append(s.fitness)
            s.adjusted_fitness = None
            if prev_fitness is None or s.fitness > prev_fitness:
                s.last_improved = generation

            species_data.append((sid, s))

        # Sort in ascending fitness order.
        species_data.sort(key=lambda x: x[1].fitness)

        result = []
        species_fitnesses = []
        num_non_stagnant = len(species_data)
        for sid, s in species_data:
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species
            # will be marked as stagnant first.
            stagnant_time = generation - s.last_improved
            is_stagnant = False
            if num_non_stagnant > self.species_elitism:
                is_stagnant = stagnant_time >= self.max_stagnation

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((sid, s, is_stagnant))
            species_fitnesses.append(s.fitness)

        return result