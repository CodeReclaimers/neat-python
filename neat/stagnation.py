import sys

from neat.species import Species
from neat.six_util import iteritems

# TODO: Add a method for the user to change the "is stagnant" computation.
# TODO: Add a mechanism to prevent stagnation of the top N species.


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
        species_data = []
        for sid, s in iteritems(species):
            fitness = self.species_fitness_func(s)
            prev_fitness = self.previous_fitnesses.get(sid, -sys.float_info.max)
            if fitness > prev_fitness:
                scount = 0
                self.previous_fitnesses[sid] = fitness
            else:
                scount = self.stagnant_counts.get(sid, 0) + 1

            self.stagnant_counts[sid] = scount

            is_stagnant = scount >= self.max_stagnation
            species_data.append((sid, s, is_stagnant, fitness))

        # Sort in ascending fitness order.
        species_data.sort(key=lambda x: x[3])

        result = []
        species_fitnesses = []
        for sid, s, is_stagnant, fitness in species_data:
            # Override stagnant state if marking this species as stagnant would
            # result in the total number of species dropping below the limit.
            # Because species are in ascending fitness order, less fit species
            # will be marked as stagnant first.
            if is_stagnant and len(self.stagnant_counts) <= self.species_elitism:
                is_stagnant = False

            # Remove stagnant species tracking data.
            if is_stagnant:
                self.remove(s)

            result.append((sid, s, is_stagnant))
            species_fitnesses.append(fitness)



        self.reporters.info('Species fitness  : {0!r}'.format(species_fitnesses))

        # TODO: shouldn't this information be a specific event type instead of just "info"?
        # TODO: this should probably be reported higher up by the caller of update().
        self.reporters.info('Species no improv: {0!r}'.format(self.stagnant_counts))

        return result