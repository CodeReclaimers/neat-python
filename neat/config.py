import os

from neat.genes import NodeGene, ConnectionGene
from neat.genome import Genome, FFGenome
from neat.nn import activations
from neat.diversity import ExplicitFitnessSharing

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import SafeConfigParser as ConfigParser


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

    allowed_activation = list(activations.keys())
    allowed_connectivity = ['unconnected', 'fs_neat', 'fully_connected', 'partial']

    def __init__(self, filename):
        if not os.path.isfile(filename):
            raise Exception('No such config file: ' + os.path.abspath(filename))

        with open(filename) as f:
            parameters = ConfigParser()
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)

        # Phenotype configuration
        self.input_nodes = int(parameters.get('phenotype', 'input_nodes'))
        self.output_nodes = int(parameters.get('phenotype', 'output_nodes'))
        self.hidden_nodes = int(parameters.get('phenotype', 'hidden_nodes'))
        self.initial_connection = parameters.get('phenotype', 'initial_connection')
        self.connection_fraction = None
        self.max_weight = float(parameters.get('phenotype', 'max_weight'))
        self.min_weight = float(parameters.get('phenotype', 'min_weight'))
        self.feedforward = bool(int(parameters.get('phenotype', 'feedforward')))
        self.weight_stdev = float(parameters.get('phenotype', 'weight_stdev'))
        self.activation_functions = parameters.get('phenotype', 'activation_functions').strip().split()

        # Verify that initial connection type is valid.
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise Exception("'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify that specified activation functions are valid.
        assert all(x in self.allowed_activation for x in self.activation_functions)

        # Select a genotype class.
        if self.feedforward:
            self.genotype = FFGenome
        else:
            self.genotype = Genome

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
        self.prob_toggle_link = float(parameters.get('genetic', 'prob_toggle_link'))
        self.elitism = int(parameters.get('genetic', 'elitism'))
        self.reset_on_extinction = bool(int(parameters.get('genetic', 'reset_on_extinction')))

        # genotype compatibility
        self.compatibility_threshold = float(parameters.get('genotype compatibility', 'compatibility_threshold'))
        self.excess_coefficient = float(parameters.get('genotype compatibility', 'excess_coefficient'))
        self.disjoint_coefficient = float(parameters.get('genotype compatibility', 'disjoint_coefficient'))
        self.weight_coefficient = float(parameters.get('genotype compatibility', 'weight_coefficient'))

        # species
        self.survival_threshold = float(parameters.get('species', 'survival_threshold'))
        self.max_stagnation = int(parameters.get('species', 'max_stagnation'))

        # Gene types
        self.node_gene_type = NodeGene
        self.conn_gene_type = ConnectionGene

        self.diversity_type=ExplicitFitnessSharing

        # Show stats after each generation.
        self.report = True
        # Save the best genome from each generation.
        self.save_best = False
        # Time in minutes between saving checkpoints, None for no timed checkpoints.
        self.checkpoint_interval = None
        # Time in generations between saving checkpoints, None for no generational checkpoints.
        self.checkpoint_generation = None
