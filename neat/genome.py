from neat.genes import ConnectionGene, NodeGene
from neat.six_util import iteritems, itervalues, iterkeys

from neat.activations import ActivationFunctionSet

from math import fabs
from random import choice, gauss, randint, random, shuffle


class DefaultGenomeConfig(object):
    allowed_connectivity = ['unconnected', 'fs_neat', 'fully_connected', 'partial']
    aggregation_function_defs = {'sum': sum, 'max': max, 'min': min}

    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()

        num_inputs = int(params.get('num_inputs', 0))
        num_outputs = int(params.get('num_outputs', 0))
        self.set_input_output_sizes(num_inputs, num_outputs)

        self.feed_forward = bool(int(params.get('feed_forward', 0)))
        self.hidden_nodes = int(params.get('num_hidden', 0))
        self.connection_fraction = None

        self.activation = params.get('activation', 'sigmoid').strip().split()

        # TODO: Verify that specified activation functions are valid before using them.
        # for fn in self.activation:
        #     if not self.activation_defs.is_valid(fn):
        #         raise Exception("Invalid activation function name: {0!r}".format(fn))

        self.aggregation = params.get('aggregation', 'sum').strip().split()

        # Verify that initial connection type is valid.
        self.initial_connection = params.get('initial_connection', 'unconnected')
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise Exception("'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Mutation parameters.
        self.activation_mutate_prob = float(params.get('activation_mutate_prob', 0.0))
        self.aggregation_mutate_prob = float(params.get('aggregation_mutate_prob', 0.0))

        self.conn_add_prob = float(params.get('conn_add_prob', 0.5))
        self.conn_delete_prob = float(params.get('conn_delete_prob', 0.1))

        self.node_add_prob = float(params.get('node_add_prob', 0.1))
        self.node_delete_prob = float(params.get('node_delete_prob', 0.05))

        self.bias_mutate_prob = float(params.get('bias_mutate_prob', 0.05))
        self.bias_mutate_power = float(params.get('bias_mutate_power', 2.0))
        self.response_mutate_prob = float(params.get('response_mutate_prob', 0.5))
        self.response_mutate_power = float(params.get('response_mutate_power', 0.1))

        self.weight_max = float(params.get('weight_max', 30.0))
        self.weight_min = float(params.get('weight_min', 30.0))
        self.weight_mean = float(params.get('weight_mean', 0.0))
        self.weight_stdev = float(params.get('weight_stdev', 1.0))
        self.weight_mutate_prob = float(params.get('weight_mutate_prob', 0.5))
        self.weight_replace_prob = float(params.get('weight_replace_prob', 0.02))
        self.weight_mutate_power = float(params.get('weight_mutate_power', 0.8))

        self.link_toggle_prob = float(params.get('link_toggle_prob', 0.01))

        # Genotype compatibility parameters.
        self.compatibility_threshold = float(params.get('compatibility_threshold', 3.0))
        self.excess_coefficient = float(params.get('excess_coefficient', 1.0))
        self.disjoint_coefficient = float(params.get('disjoint_coefficient', 1.0))
        self.weight_coefficient = float(params.get('weight_coefficient', 0.4))

    def set_input_output_sizes(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

    # TODO: Factor out these mutation methods into a separate class?
    def new_weight(self):
        return gauss(0, self.weight_stdev)

    def new_bias(self):
        return gauss(0, self.weight_stdev)

    def new_response(self):
        return 5.0

    def new_aggregation(self):
        return choice(self.aggregation)

    def new_activation(self):
        return choice(self.activation)

    def mutate_weight(self, weight):
        if random() < self.weight_mutate_prob:
            if random() < self.weight_replace_prob:
                # Replace weight with a random value.
                weight = self.new_weight()
            else:
                # Perturb weight.
                weight += gauss(0, self.weight_mutate_power)
                weight = max(self.weight_min, min(self.weight_max, weight))

        return weight

    def mutate_bias(self, bias):
        if random() < self.bias_mutate_prob:
            bias += gauss(0, self.bias_mutate_power)
            bias = max(self.weight_min, min(self.weight_max, bias))

        return bias

    def mutate_response(self, response):
        if random() < self.response_mutate_prob:
            response += gauss(0, self.response_mutate_power)
            response = max(self.weight_min, min(self.weight_max, response))

        return response

    def mutate_aggregation(self, aggregation):
        if random() < self.aggregation_mutate_prob:
            aggregation = self.new_aggregation()

        return aggregation

    def mutate_activation(self, activation):
        if random() < self.activation_mutate_prob:
            activation = self.new_activation()

        return activation


class DefaultGenome(object):
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node:
        connection:
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

    @staticmethod
    def create_config(kwargs):
        return DefaultGenomeConfig(kwargs)

    def __init__(self, key, config):
        """
        :param key: This genome's unique identifier.
        :param config: A neat.config.Config instance.
        """
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}

        # Fitness results.
        # TODO: This should probably be stored elsewhere.
        self.fitness = None
        self.cross_fitness = None

    def add_node(self, key, bias, response, aggregation, activation):
        # TODO: Add validation of this node addition.
        self.nodes[key] = NodeGene(key, bias, response, aggregation, activation)

    def add_connection(self, input_key, output_key, weight, enabled):
        # TODO: Add validation of this connection addition.
        self.connections[input_key, output_key] = ConnectionGene(input_key, output_key, weight, enabled)

    def mutate(self, config):
        """ Mutates this genome. """

        # TODO: Make a configuration item to choose whether or not multiple
        # mutations can happen simulataneously.
        genome_config = config.genome_config
        if random() < genome_config.node_add_prob:
            self.mutate_add_node(config)

        if random() < genome_config.node_delete_prob:
            self.mutate_delete_node(config)

        if random() < genome_config.conn_add_prob:
            self.mutate_add_connection(config)

        if random() < genome_config.conn_delete_prob:
            self.mutate_delete_connection()

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    def crossover(self, other, key, config):
        """ Crosses over parents' genomes and returns a child. """
        if self.fitness > other.fitness:
            parent1 = self
            parent2 = other
        else:
            parent1 = other
            parent2 = self

        # creates a new child
        child = DefaultGenome(key, config)
        child.inherit_genes(parent1, parent2)

        return child

    def inherit_genes(self, parent1, parent2):
        """ Applies the crossover operator. """
        assert (parent1.fitness >= parent2.fitness)

        # Inherit connection genes
        for key, cg1 in iteritems(parent1.connections):
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

    def get_new_hidden_id(self):
        new_id = 0
        while new_id in self.nodes:
            new_id += 1
        return new_id

    def mutate_add_node(self, config):
        if not self.connections:
            return None, None

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = self.get_new_hidden_id()
        act_func = choice(config.genome_config.activation)
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng
        new_conn1, new_conn2 = conn_to_split.split(new_node_id)
        self.connections[new_conn1.key] = new_conn1
        self.connections[new_conn2.key] = new_conn2
        return ng, conn_to_split  # the return is only used in genome_feedforward

    def mutate_add_connection(self, config):
        '''
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        '''
        possible_outputs = list(iterkeys(self.nodes))
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.genome_config.input_keys
        in_node = choice(possible_inputs)

        # Only create the connection if it doesn't already exist.
        key = (in_node, out_node)
        if key not in self.connections:
            # TODO: factor out new connection creation based on config
            weight = gauss(0, config.genome_config.weight_stdev)
            enabled = choice([False, True])
            cg = ConnectionGene(in_node, out_node, weight, enabled)
            self.connections[cg.key] = cg

    def mutate_delete_node(self, config):
        # Do nothing if there are no non-output nodes.
        available_nodes = [(k, v) for k, v in iteritems(self.nodes) if k not in config.genome_config.output_keys]
        if not available_nodes:
            return -1

        del_key, del_node = choice(available_nodes)

        connections_to_delete = set()
        for k, v in self.connections.items():
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        return del_key

    def mutate_delete_connection(self):
        if self.connections:
            key = choice(list(self.connections.keys()))
            del self.connections[key]

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """
        genome_config = config.genome_config

        # Take genome1 to be the one with the most connections.
        genome1, genome2 = self, other
        if len(self.connections) > len(other.connections):
            genome1, genome2 = self, other
        else:
            genome2, genome1 = self, other

        node_genes1 = genome1.nodes
        node_gene_count1 = len(node_genes1)

        node_genes2 = genome2.nodes
        node_gene_count2 = len(node_genes2)

        # Compute node gene differences.
        excess1 = 0
        excess2 = 0
        bias_diff = 0.0
        response_diff = 0.0
        activation_diff = 0
        num_common = 0


        for k2 in node_genes2.keys():
            if k2 not in node_genes1.keys():
                excess2 += 1

        for k1, g1 in iteritems(node_genes1):
            if k1 in node_genes2:
                num_common += 1
                g2 = node_genes2[k1]
                bias_diff += fabs(g1.bias - g2.bias)
                response_diff += fabs(g1.response - g2.response)
                if g1.activation != g2.activation:
                    activation_diff += 1
            else:
                excess1 += 1

        most_nodes = max(node_gene_count1, node_gene_count2)
        distance = (genome_config.excess_coefficient * float(excess1 + excess2) / most_nodes
                    + genome_config.excess_coefficient * float(activation_diff) / most_nodes
                    + genome_config.weight_coefficient * (bias_diff + response_diff) / num_common)

        # Compute connection gene differences.
        if genome1.connections:
            N = len(genome1.connections)
            weight_diff = 0
            matching = 0
            disjoint = 0
            excess = 0

            max_cg_genome2 = None
            if genome2.connections:
                max_cg_genome2 = max(itervalues(genome2.connections))

            for k1, cg1 in iteritems(genome1.connections):
                if k1 in genome2.connections:
                    # Homologous genes
                    cg2 = genome2.connections[k1]
                    weight_diff += fabs(cg1.weight - cg2.weight)
                    matching += 1

                    if cg1.enabled != cg2.enabled:
                        weight_diff += 1.0
                else:
                    if max_cg_genome2 is not None and cg1 > max_cg_genome2:
                        excess += 1
                    else:
                        disjoint += 1

            disjoint += len(genome2.connections) - matching

            distance += genome_config.excess_coefficient * float(excess) / N
            distance += genome_config.disjoint_coefficient * float(disjoint) / N
            if matching > 0:
                distance += genome_config.weight_coefficient * (weight_diff / matching)

        compatible = distance < genome_config.compatibility_threshold

        return distance, compatible

    def size(self, config):
        '''Returns genome 'complexity', taken to be (number of hidden nodes, number of enabled connections)'''
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled is True])
        return len(self.nodes) - len(config.genome_config.output_keys), num_enabled_connections

    def __str__(self):
        s = "Nodes:"
        for k, ng in iteritems(self.nodes):
            s += "{0} {1!s}\n\t".format(k, ng)
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s

    def add_hidden_nodes(self, num_hidden, config):
        node_id = self.get_new_hidden_id()
        for i in range(num_hidden):
            # TODO: factor out new node creation.
            act_func = choice(config.activation)
            node_gene = config.node_gene_type(activation_type=act_func)
            assert node_id not in self.hidden
            self.hidden[node_id] = node_gene
            node_id += 1

    @classmethod
    def create(cls, config, key):
        genome_config = config.genome_config
        g = cls.create_unconnected(config, key)

        # Add hidden nodes if requested.
        if genome_config.hidden_nodes > 0:
            g.add_hidden_nodes(genome_config.hidden_nodes)

        # Add connections based on initial connectivity type.
        if genome_config.initial_connection == 'fs_neat':
            g.connect_fs_neat()
        elif genome_config.initial_connection == 'fully_connected':
            g.connect_full(config)
        elif genome_config.initial_connection == 'partial':
            g.connect_partial(config)

        return g

    @staticmethod
    def create_node(config, node_id):
        genome_config = config.genome_config
        return NodeGene(node_id, genome_config.new_bias(), genome_config.new_response(),
                        genome_config.new_aggregation(), genome_config.new_activation())

    @classmethod
    def create_unconnected(cls, config, key):
        '''Create a genome for a network with no hidden nodes and no connections.'''
        c = cls(key, config)

        # Create node genes for the output pins.
        for node_key in config.genome_config.output_keys:
            c.nodes[node_key] = cls.create_node(config, node_key)

        return c

    def connect_fs_neat(self, config):
        """ Randomly connect one input to all hidden and output nodes (FS-NEAT). """
        genome_config = config.genome_config

        # TODO: Factor out the gene creation.
        input_id = choice(self.inputs.keys())
        for output_id in list(self.hidden.keys()) + list(self.outputs.keys()):
            weight = gauss(0, genome_config.weight_stdev)
            cg = ConnectionGene(input_id, output_id, weight, True)
            self.connections[cg.key] = cg

    def compute_full_connections(self, config):
        """ Compute connections for a fully-connected genome (each input connected to all nodes). """
        genome_config = config.genome_config

        connections = []
        for input_id in genome_config.input_keys:
            for node_id in iterkeys(self.nodes):
                connections.append((input_id, node_id))

        return connections

    def connect_full(self, config):
        """ Create a fully-connected genome. """
        genome_config = config.genome_config

        # TODO: Factor out the gene creation.
        for input_id, output_id in self.compute_full_connections(config):
            weight = gauss(0, genome_config.weight_stdev)
            cg = ConnectionGene(input_id, output_id, weight, True)
            self.connections[cg.key] = cg

    def connect_partial(self, config):
        genome_config = config.genome_config

        # TODO: Factor out the gene creation.
        assert 0 <= genome_config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * genome_config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            weight = gauss(0, genome_config.weight_stdev)
            cg = ConnectionGene(input_id, output_id, weight, True)
            self.connections[cg.key] = cg
