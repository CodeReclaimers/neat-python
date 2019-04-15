"""Keeps track of whether species are making progress and helps remove ones that are not."""
import sys

from neat.config import ConfigParameter, DefaultClassConfig
from neat.six_util import iteritems
from neat.math_util import stat_functions


class DefaultStagnation:
    """Keeps track of whether species are making progress and helps remove ones that are not."""

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('species_fitness_func', str, 'mean'),
                                   ConfigParameter('max_stagnation', int, 15),
                                   ConfigParameter('species_elitism', int, 0)])

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.stagnation_config = config

        self.species_fitness_func = stat_functions.get(config.species_fitness_func)
        if self.species_fitness_func is None:
            raise RuntimeError(
                "Unexpected species fitness func: {0!r}".format(config.species_fitness_func))

        self.reporters = reporters

    def update(self, species_set, generation):
        """
        Required interface method. Updates species fitness history information,
        checking for ones that have not improved in max_stagnation generations,
        and - unless it would result in the number of species dropping below the configured
        species_elitism parameter if they were removed,
        in which case the highest-fitness species are spared -
        returns a list with stagnant species marked for removal.
        """
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
        num_stagnant = 0
        for idx, (sid, s) in enumerate(species_data):

            is_stagnant = self.check_stagnant(idx, s, len(species_data), num_stagnant, generation)

            if is_stagnant:
                num_stagnant += 1

            result.append((sid, s, is_stagnant))
            species_fitnesses.append(s.fitness)

        return result

    def check_stagnant(self, idx, species, num_species, num_stagnant, generation):
        """
        This function checks whether a species is stagnant and returns true if so, false otherwise.
        :param idx          - the location of the species when species are sorted on fitness in ascending order.
        :param species      - The species itself.
        :param num_species  - The current number of species.
        :param num_stagnant - The number of species marked stagnant so far.
        :param generation   - The number of generations this far.
        """
        stagnant_time = generation - species.last_improved

        # Override stagnant state if marking this species as stagnant would
        # result in the total number of species dropping below the limit.
        # Because species are in ascending fitness order, less fit species
        # will be marked as stagnant first.

        is_stagnant = False
        if num_species - num_stagnant > self.stagnation_config.species_elitism:
            is_stagnant = stagnant_time >= self.stagnation_config.max_stagnation

        if (num_species - idx) <= self.stagnation_config.species_elitism:
            is_stagnant = False

        return is_stagnant


class MarkAllStagnation(DefaultStagnation):
    """
    This class marks all species stagnated that did not improve for a number of generations that is dividable by
    stagnation_split_interval.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('species_fitness_func', str, 'mean'),
                                   ConfigParameter('stagnation_split_interval', int, 15)])

    def check_stagnant(self, idx, species, num_species, num_stagnant, generation):

        stagnant_time = generation - species.last_improved

        return stagnant_time != 0 and stagnant_time % self.stagnation_config.stagnation_split_interval == 0
