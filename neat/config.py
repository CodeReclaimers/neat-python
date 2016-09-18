import os

from neat.genome import DefaultGenome
from neat.reproduction import DefaultReproduction
from neat.stagnation import DefaultStagnation

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import SafeConfigParser as ConfigParser


class Config(object):
    '''
    A simple container for all of the user-configurable parameters of NEAT.
    '''

    def __init__(self, filename=None, reg_dict={}):
        # Initialize type registry with default implementations.
        self.registry = {'DefaultStagnation': DefaultStagnation,
                         'DefaultReproduction': DefaultReproduction,
                         'DefaultGenome': DefaultGenome}
        self.registry.update(reg_dict)

        genome_type_name = 'DefaultGenome'
        genome_config = {}

        reproduction_type_name = 'DefaultReproduction'
        reproduction_config = {}

        stagnation_type_name = 'DefaultStagnation'
        stagnation_config = {}


        parameters = ConfigParser()
        if filename is not None:
            if not os.path.isfile(filename):
                raise Exception('No such config file: ' + os.path.abspath(filename))

            with open(filename) as f:
                if hasattr(parameters, 'read_file'):
                    parameters.read_file(f)
                else:
                    parameters.readfp(f)

            # NEAT configuration
            if not parameters.has_section('NEAT'):
                raise RuntimeError("'NEAT' section not found in NEAT configuration file.")

            # Type registry.
            if not parameters.has_section('Types'):
                raise RuntimeError("'Types' section not found in NEAT configuration file.")

            genome_type_name = parameters.get('Types', 'genome_type')
            genome_config = dict(parameters.items(genome_type_name))

            reproduction_type_name = parameters.get('Types', 'reproduction_type')
            reproduction_config = dict(parameters.items(reproduction_type_name))

            stagnation_type_name = parameters.get('Types', 'stagnation_type')
            stagnation_config = dict(parameters.items(stagnation_type_name))


        self.pop_size = int(parameters.get('NEAT', 'pop_size', fallback=150))
        self.max_fitness_threshold = float(parameters.get('NEAT', 'max_fitness_threshold', fallback=-0.05))
        self.reset_on_extinction = bool(int(parameters.get('NEAT', 'reset_on_extinction', fallback=False)))

        # Gather statistics for each generation.
        self.collect_statistics = True
        # Show stats after each generation.
        self.report = True
        # Save the best genome from each generation.
        self.save_best = False
        # Time in minutes between saving checkpoints, None for no timed checkpoints.
        self.checkpoint_time_interval = None
        # Time in generations between saving checkpoints, None for no generational checkpoints.
        self.checkpoint_gen_interval = None


        # Genome type configuration.
        if genome_type_name not in self.registry:
            raise Exception('Unknown genome type: {!r}'.format(genome_type_name))
        self.genome_type = self.registry[genome_type_name]
        self.genome_config = self.genome_type.create_config(genome_config)

        # Reproduction type configuration.
        if reproduction_type_name not in self.registry:
            raise Exception('Unknown reproduction type: {!r}'.format(reproduction_type_name))
        self.reproduction_type = self.registry[reproduction_type_name]
        self.reproduction_config = self.reproduction_type.create_config(reproduction_config)

        # Stagnation type configuration.
        if stagnation_type_name not in self.registry:
            raise Exception('Unknown stagnation type: {!r}'.format(stagnation_type_name))
        self.stagnation_type = self.registry[stagnation_type_name]
        self.stagnation_config = self.stagnation_type.create_config(stagnation_config)

    def save(self, filename):
        # TODO: Implement
        pass

    def register(self, type_name, type_def):
        """
        User-defined classes mentioned in the config file must be provided to the
        configuration object before the load() method is called.
        """
        self.registry[type_name] = type_def

