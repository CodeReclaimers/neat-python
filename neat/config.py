import os
from random import random, gauss, choice

from neat.activations import ActivationFunctionSet
from neat.genome import DefaultGenome
from neat.reproduction import DefaultReproduction
from neat.stagnation import DefaultStagnation

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import SafeConfigParser as ConfigParser

aggregation_function_defs = {'sum': sum, 'max': max, 'min': min}


class Config(object):
    '''
    A simple container for all of the user-configurable parameters of NEAT.
    '''

    # TODO: Add the ability to write a Config to text file.

    # TODO: Split out the configuration into implementation-specific sections. For example,
    # a node gene class FooNode would expect to find a [FooNode] section within the configuration
    # file, and the NEAT framework doesn't need to know about this section in any way. This
    # allows all the configuration to stay in one text file, without unnecessary complication.
    # It also makes the config file and associated setup code somewhat self-documenting, as the
    # classes you need to give to NEAT are shown in the config file.

    allowed_connectivity = ['unconnected', 'fs_neat', 'fully_connected', 'partial']

    def __init__(self):
        # Initialize type registry with default implementations.
        self.registry = {'DefaultStagnation': DefaultStagnation,
                         'DefaultReproduction': DefaultReproduction,
                         'DefaultGenome': DefaultGenome}
        self.type_config = {}

        # Phenotype configuration
        self.input_nodes = 0
        self.output_nodes = 0
        self.hidden_nodes = 0
        self.initial_connection = 'unconnected'
        self.connection_fraction = None
        self.max_weight = 30.0
        self.min_weight = -30.0
        self.weight_stdev = 1.0
        self.activation_functions = ['sigmoid']
        self.aggregation_functions = ['sum']

        # Genetic algorithm configuration
        self.pop_size = 150
        self.max_fitness_threshold = -0.05
        self.prob_add_conn = 0.5
        self.prob_add_node = 0.1
        self.prob_delete_conn = 0.1
        self.prob_delete_node = 0.05
        self.prob_mutate_bias = 0.05
        self.bias_mutation_power = 2.0
        self.prob_mutate_response = 0.5
        self.response_mutation_power = 0.1
        self.prob_mutate_weight = 0.5
        self.prob_replace_weight = 0.02
        self.weight_mutation_power = 0.8
        self.prob_mutate_activation = 0.0
        self.prob_mutate_aggregation = 0.0
        self.prob_toggle_link = 0.01
        self.reset_on_extinction = True

        # genotype compatibility
        self.compatibility_threshold = 3.0
        self.excess_coefficient = 1.0
        self.disjoint_coefficient = 1.0
        self.weight_coefficient = 0.4

        stagnation_type_name = 'DefaultStagnation'
        self.stagnation_type = self.registry[stagnation_type_name]
        # TODO: Look up the default type configuration from a static method on the type?
        self.type_config[stagnation_type_name] = {'species_fitness_func': 'mean',
                                                  'max_stagnation': 15}

        reproduction_type_name = 'DefaultReproduction'
        self.reproduction_type = self.registry[reproduction_type_name]
        # TODO: Look up the default type configuration from a static method on the type?
        self.type_config[reproduction_type_name] = {'elitism': 1,
                                                    'survival_threshold': 0.2}

        genome_type_name = 'DefaultGenome'
        self.genome_type = self.registry[genome_type_name]
        # TODO: Look up the default type configuration from a static method on the type?
        self.type_config[genome_type_name] = {}

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

        # Create full set of available activation functions.
        # TODO: pick a better name for this member, it's too confusing alongside activation_functions.
        self.available_activations = ActivationFunctionSet()

    def load(self, filename):
        if not os.path.isfile(filename):
            raise Exception('No such config file: ' + os.path.abspath(filename))

        with open(filename) as f:
            parameters = ConfigParser()
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)

        if not parameters.has_section('Types'):
            raise RuntimeError("'Types' section not found in NEAT configuration file.")

        # Phenotype configuration
        self.input_nodes = int(parameters.get('phenotype', 'input_nodes'))
        self.output_nodes = int(parameters.get('phenotype', 'output_nodes'))
        self.hidden_nodes = int(parameters.get('phenotype', 'hidden_nodes'))
        self.initial_connection = parameters.get('phenotype', 'initial_connection')
        self.connection_fraction = None
        self.max_weight = float(parameters.get('phenotype', 'max_weight'))
        self.min_weight = float(parameters.get('phenotype', 'min_weight'))
        self.weight_stdev = float(parameters.get('phenotype', 'weight_stdev'))
        self.activation_functions = parameters.get('phenotype', 'activation_functions').strip().split()
        self.aggregation_functions = parameters.get('phenotype', 'aggregation_functions').strip().split()

        # Verify that initial connection type is valid.
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise Exception("'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify that specified activation functions are valid.
        for fn in self.activation_functions:
            if not self.available_activations.is_valid(fn):
                raise Exception("Invalid activation function name: {0!r}".format(fn))

        # Genetic algorithm configuration
        self.pop_size = int(parameters.get('genetic', 'pop_size'))
        self.max_fitness_threshold = float(parameters.get('genetic', 'max_fitness_threshold'))
        self.prob_add_conn = float(parameters.get('genetic', 'prob_add_conn'))
        self.prob_add_node = float(parameters.get('genetic', 'prob_add_node'))
        self.prob_delete_conn = float(parameters.get('genetic', 'prob_delete_conn'))
        self.prob_delete_node = float(parameters.get('genetic', 'prob_delete_node'))
        self.prob_mutate_bias = float(parameters.get('genetic', 'prob_mutate_bias'))
        self.bias_mutation_power = float(parameters.get('genetic', 'bias_mutation_power'))
        self.prob_mutate_response = float(parameters.get('genetic', 'prob_mutate_response'))
        self.response_mutation_power = float(parameters.get('genetic', 'response_mutation_power'))
        self.prob_mutate_weight = float(parameters.get('genetic', 'prob_mutate_weight'))
        self.prob_replace_weight = float(parameters.get('genetic', 'prob_replace_weight'))
        self.weight_mutation_power = float(parameters.get('genetic', 'weight_mutation_power'))
        self.prob_mutate_activation = float(parameters.get('genetic', 'prob_mutate_activation'))
        self.prob_mutate_aggregation = float(parameters.get('genetic', 'prob_mutate_aggregation'))
        self.prob_toggle_link = float(parameters.get('genetic', 'prob_toggle_link'))
        self.reset_on_extinction = bool(int(parameters.get('genetic', 'reset_on_extinction')))

        # genotype compatibility
        self.compatibility_threshold = float(parameters.get('genotype compatibility', 'compatibility_threshold'))
        self.excess_coefficient = float(parameters.get('genotype compatibility', 'excess_coefficient'))
        self.disjoint_coefficient = float(parameters.get('genotype compatibility', 'disjoint_coefficient'))
        self.weight_coefficient = float(parameters.get('genotype compatibility', 'weight_coefficient'))

        stagnation_type_name = parameters.get('Types', 'stagnation_type')
        reproduction_type_name = parameters.get('Types', 'reproduction_type')
        genome_type_name = parameters.get('Types', 'genome_type')

        if stagnation_type_name not in self.registry:
            raise Exception('Unknown stagnation type: {!r}'.format(stagnation_type_name))
        self.stagnation_type = self.registry[stagnation_type_name]
        self.type_config[stagnation_type_name] = parameters.items(stagnation_type_name)

        if reproduction_type_name not in self.registry:
            raise Exception('Unknown reproduction type: {!r}'.format(reproduction_type_name))
        self.reproduction_type = self.registry[reproduction_type_name]
        self.type_config[reproduction_type_name] = parameters.items(reproduction_type_name)

        if genome_type_name not in self.registry:
            raise Exception('Unknown reproduction type: {!r}'.format(reproduction_type_name))
        self.genome_type = self.registry[genome_type_name]
        self.type_config[genome_type_name] = parameters.items(genome_type_name)

    def set_input_output_sizes(self, num_inputs, num_outputs):
        self.input_nodes = num_inputs
        self.output_nodes = num_outputs
        self.input_keys = [-i-1 for i in range(self.input_nodes)]
        self.output_keys = [i for i in range(self.output_nodes)]

    def save(self, filename):
        pass

    def register(self, type_name, type_def):
        """
        User-defined classes mentioned in the config file must be provided to the
        configuration object before the load() method is called.
        """
        self.registry[type_name] = type_def

    def get_type_config(self, type_instance):
        return dict(self.type_config[type_instance.__class__.__name__])

    # TODO: Factor out these mutation methods into a separate class?
    def new_weight(self):
        return gauss(0, self.weight_stdev)

    def new_bias(self):
        return gauss(0, self.weight_stdev)

    def new_response(self):
        return 5.0

    def new_aggregation(self):
        return choice(self.aggregation_functions)

    def new_activation(self):
        return choice(self.activation_functions)

    def mutate_weight(self, weight):
        if random() < self.prob_mutate_weight:
            if random() < self.prob_replace_weight:
                # Replace weight with a random value.
                weight = self.new_weight()
            else:
                # Perturb weight.
                weight += gauss(0, self.weight_mutation_power)
                weight = max(self.min_weight, min(self.max_weight, weight))

        return weight

    def mutate_bias(self, bias):
        if random() < self.prob_mutate_bias:
            bias += gauss(0, self.bias_mutation_power)
            bias = max(self.min_weight, min(self.max_weight, bias))

        return bias

    def mutate_response(self, response):
        if random() < self.prob_mutate_response:
            response += gauss(0, self.response_mutation_power)
            response = max(self.min_weight, min(self.max_weight, response))

        return response

    def mutate_aggregation(self, aggregation):
        if random() < self.prob_mutate_aggregation:
            aggregation = self.new_aggregation()

        return aggregation

    def mutate_activation(self, activation):
        if random() < self.prob_mutate_activation:
            activation = self.new_activation()

        return activation
