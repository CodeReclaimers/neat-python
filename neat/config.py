from ConfigParser import SafeConfigParser
import os


class Config(object):
    '''
    A simple container for all of the user-configurable parameters of NEAT.
    '''

    def __init__(self, filename):
        if not os.path.isfile(filename):
            raise Exception('No such config file: ' + os.path.abspath(filename))

        with open(filename) as f:
            parameters = SafeConfigParser()
            parameters.readfp(f)

        # Phenotype configuration
        self.input_nodes = int(parameters.get('phenotype', 'input_nodes'))
        self.output_nodes = int(parameters.get('phenotype', 'output_nodes'))
        self.hidden_nodes = int(parameters.get('phenotype', 'hidden_nodes'))
        self.fully_connected = bool(int(parameters.get('phenotype', 'fully_connected')))
        self.max_weight = float(parameters.get('phenotype', 'max_weight'))
        self.min_weight = float(parameters.get('phenotype', 'min_weight'))
        self.feedforward = bool(int(parameters.get('phenotype', 'feedforward')))
        self.nn_activation = parameters.get('phenotype', 'nn_activation')  # exp or tanh
        self.weight_stdev = float(parameters.get('phenotype', 'weight_stdev'))

        # Genetic algorithm configuration
        self.pop_size = int(parameters.get('genetic', 'pop_size'))
        self.max_fitness_threshold = float(parameters.get('genetic', 'max_fitness_threshold'))
        self.prob_addconn = float(parameters.get('genetic', 'prob_addconn'))
        self.prob_addnode = float(parameters.get('genetic', 'prob_addnode'))
        self.prob_deleteconn = float(parameters.get('genetic', 'prob_deleteconn'))
        self.prob_deletenode = float(parameters.get('genetic', 'prob_deletenode'))
        self.prob_mutate_bias = float(parameters.get('genetic', 'prob_mutate_bias'))
        self.bias_mutation_power = float(parameters.get('genetic', 'bias_mutation_power'))
        self.prob_mutate_response = float(parameters.get('genetic', 'prob_mutate_response'))
        self.response_mutation_power = float(parameters.get('genetic', 'response_mutation_power'))
        self.prob_mutate_weight = float(parameters.get('genetic', 'prob_mutate_weight'))
        self.prob_replace_weight = float(parameters.get('genetic', 'prob_replace_weight'))
        self.weight_mutation_power = float(parameters.get('genetic', 'weight_mutation_power'))
        self.prob_togglelink = float(parameters.get('genetic', 'prob_togglelink'))
        self.elitism = float(parameters.get('genetic', 'elitism'))

        # genotype compatibility
        self.compatibility_threshold = float(parameters.get('genotype compatibility', 'compatibility_threshold'))
        self.compatibility_change = float(parameters.get('genotype compatibility', 'compatibility_change'))
        self.excess_coefficient = float(parameters.get('genotype compatibility', 'excess_coefficient'))
        self.disjoint_coefficient = float(parameters.get('genotype compatibility', 'disjoint_coefficient'))
        self.weight_coefficient = float(parameters.get('genotype compatibility', 'weight_coefficient'))

        # species
        self.species_size = int(parameters.get('species', 'species_size'))
        self.survival_threshold = float(parameters.get('species', 'survival_threshold'))
        self.old_threshold = int(parameters.get('species', 'old_threshold'))
        self.youth_threshold = int(parameters.get('species', 'youth_threshold'))
        self.old_penalty = float(parameters.get('species', 'old_penalty'))
        self.youth_boost = float(parameters.get('species', 'youth_boost'))
        self.max_stagnation = int(parameters.get('species', 'max_stagnation'))
