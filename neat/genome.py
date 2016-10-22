from neat.config import ConfigParameter, write_pretty_params
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.six_util import iteritems, itervalues, iterkeys

from neat.activations import ActivationFunctionSet
from neat.nn import creates_cycle

from math import fabs
from random import choice, random, shuffle


class DefaultGenomeConfig(object):
    __params = [ConfigParameter('num_inputs', int),
                ConfigParameter('num_outputs', int),
                ConfigParameter('num_hidden', int),
                ConfigParameter('feed_forward', bool),
                ConfigParameter('compatibility_threshold', float),
                ConfigParameter('excess_coefficient', float),
                ConfigParameter('disjoint_coefficient', float),
                ConfigParameter('weight_coefficient', float),
                ConfigParameter('conn_add_prob', float),
                ConfigParameter('conn_delete_prob', float),
                ConfigParameter('node_add_prob', float),
                ConfigParameter('node_delete_prob', float)]

    allowed_connectivity = ['unconnected', 'fs_neat', 'fully_connected', 'partial']
    aggregation_function_defs = {'sum': sum, 'max': max, 'min': min}

    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        self.activation_options = params.get('activation_options', 'sigmoid').strip().split()

        # TODO: Verify that specified activation functions are valid before using them.
        # for fn in self.activation:
        #     if not self.activation_defs.is_valid(fn):
        #         raise Exception("Invalid activation function name: {0!r}".format(fn))

        self.aggregation_options = params.get('aggregation_options', 'sum').strip().split()

        # Gather configuration data from the gene classes.
        self.__params += DefaultNodeGene.get_config_params()
        self.__params += DefaultConnectionGene.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self.__params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        self.initial_connection = params.get('initial_connection', 'unconnected')
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise Exception("'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

    def save(self, f):
        # f.write('initial_connection      = {0}\n'.format(self.initial_connection))
        # Verify that initial connection type is valid.
        # self.initial_connection = params.get('', 'unconnected')
        # if 'partial' in self.initial_connection:
        #     c, p = self.initial_connection.split()
        #     self.initial_connection = c
        #     self.connection_fraction = float(p)
        #     if not (0 <= self.connection_fraction <= 1):
        #         raise Exception("'partial' connection value must be between 0.0 and 1.0, inclusive.")
        #
        # assert self.initial_connection in self.allowed_connectivity
        write_pretty_params(f, self, self.__params)


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

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

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

    def mutate(self, config):
        """ Mutates this genome. """

        # TODO: Make a configuration item to choose whether or not multiple
        # mutations can happen simulataneously.
        if random() < config.node_add_prob:
            self.mutate_add_node(config)

        if random() < config.node_delete_prob:
            self.mutate_delete_node(config)

        if random() < config.conn_add_prob:
            self.mutate_add_connection(config)

        if random() < config.conn_delete_prob:
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
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id, 1.0, True)
        self.add_connection(config, new_node_id, o, conn_to_split.weight, True)

    def add_connection(self, config, input_key, output_key, weight, enabled):
        # TODO: Add validation of this connection addition.
        key = (input_key, output_key)
        connection = DefaultConnectionGene(key)
        connection.init_attributes(config)
        connection.weight = weight
        connection.enabled = enabled
        self.connections[key] = connection

    def mutate_add_connection(self, config):
        '''
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        '''
        possible_outputs = list(iterkeys(self.nodes))
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key not in self.connections:
            return

        # For feed-forward networks, avoid creating cycles.
        if config.feed_forward and creates_cycle(list(iterkeys(self.connections)), key):
            return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    def mutate_delete_node(self, config):
        # Do nothing if there are no non-output nodes.
        available_nodes = [(k, v) for k, v in iteritems(self.nodes) if k not in config.output_keys]
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

        # TODO: Factor out the gene-specific distance components into the gene classes.

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
        distance = (config.excess_coefficient * float(excess1 + excess2) / most_nodes
                    + config.excess_coefficient * float(activation_diff) / most_nodes
                    + config.weight_coefficient * (bias_diff + response_diff) / num_common)

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

            distance += config.excess_coefficient * float(excess) / N
            distance += config.disjoint_coefficient * float(disjoint) / N
            if matching > 0:
                distance += config.weight_coefficient * (weight_diff / matching)

        compatible = distance < config.compatibility_threshold

        return distance, compatible

    def size(self):
        '''Returns genome 'complexity', taken to be (number of nodes, number of enabled connections)'''
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled is True])
        return len(self.nodes), num_enabled_connections

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
        g = cls.create_unconnected(config, key)

        # Add hidden nodes if requested.
        if config.num_hidden > 0:
            g.add_hidden_nodes(config.num_hidden)

        # Add connections based on initial connectivity type.
        if config.initial_connection == 'fs_neat':
            g.connect_fs_neat()
        elif config.initial_connection == 'fully_connected':
            g.connect_full(config)
        elif config.initial_connection == 'partial':
            g.connect_partial(config)

        return g

    @staticmethod
    def create_node(config, node_id):
        node = DefaultNodeGene(node_id)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = DefaultConnectionGene((input_id, output_id))
        connection.init_attributes(config)
        return connection

    @classmethod
    def create_unconnected(cls, config, key):
        '''Create a genome for a network with no hidden nodes and no connections.'''
        c = cls(key, config)

        # Create node genes for the output pins.
        for node_key in config.output_keys:
            c.nodes[node_key] = cls.create_node(config, node_key)

        return c

    def connect_fs_neat(self, config):
        """ Randomly connect one input to all hidden and output nodes (FS-NEAT). """

        # TODO: Factor out the gene creation.
        input_id = choice(self.inputs.keys())
        for output_id in list(self.hidden.keys()) + list(self.outputs.keys()):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def compute_full_connections(self, config):
        """ Compute connections for a fully-connected feed-forward genome (each input connected to all nodes). """
        connections = []
        for input_id in config.input_keys:
            for node_id in iterkeys(self.nodes):
                connections.append((input_id, node_id))

        return connections

    def connect_full(self, config):
        """ Create a fully-connected genome. """
        for input_id, output_id in self.compute_full_connections(config):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial(self, config):
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection
