from neat.genes import ConnectionGene, NodeGene
from neat.six_util import iteritems, itervalues

from math import fabs
from random import choice, gauss, randint, random, shuffle


class DefaultGenome(object):
    """ A genome for generalized neural networks. """

    def __init__(self, key):
        self.key = key

        # (id, gene) pairs for gene sets.
        self.connections = {}
        self.hidden = {}
        self.inputs = {}
        self.outputs = {}

        # TODO: Do we really need to track this, or is it sufficient
        # for each species to know which genomes are its members?
        #self.species_id = None

        # Fitness results.
        # TODO: This should probably be stored elsewhere.
        self.fitness = None
        self.cross_fitness = None

    def mutate(self, config):
        """ Mutates this genome. """

        # TODO: Make a configuration item to choose whether or not multiple mutations can happen at once.

        if random() < config.prob_add_node:
            self.mutate_add_node(config)

        if random() < config.prob_add_conn:
            self.mutate_add_connection(config)

        if random() < config.prob_delete_node:
            self.mutate_delete_node()

        if random() < config.prob_delete_conn:
            self.mutate_delete_connection()

        # Mutate connections.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate hidden and output node genes (bias, response, etc.).
        for ng in self.hidden.values():
            ng.mutate(config)
        for ng in self.outputs.values():
            ng.mutate(config)

    def crossover(self, other, key):
        """ Crosses over parents' genomes and returns a child. """
        if self.fitness > other.fitness:
            parent1 = self
            parent2 = other
        else:
            parent1 = other
            parent2 = self

        # creates a new child
        child = self.__class__(key)
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
        for set_name in ('inputs', 'hidden', 'outputs'):
            parent1_set = getattr(parent1, set_name)
            parent2_set = getattr(parent2, set_name)
            self_set = getattr(self, set_name)

            for key, ng1 in parent1_set.items():
                ng2 = parent2_set.get(key)
                assert key not in self_set
                if ng2 is None:
                    # Extra gene: copy from the fittest parent
                    self_set[key] = ng1.copy()
                else:
                    # Homologous gene: combine genes from both parents.
                    self_set[key] = ng1.crossover(ng2)

    def get_new_hidden_id(self):
        new_id = 0
        while new_id in self.inputs or new_id in self.hidden or new_id in self.outputs:
            new_id += 1
        return new_id

    def mutate_add_node(self, config):
        if not self.connections:
            return None, None

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = self.get_new_hidden_id()
        act_func = choice(config.activation_functions)
        ng = self.create_node(config, new_node_id)
        self.hidden[new_node_id] = ng
        new_conn1, new_conn2 = conn_to_split.split(new_node_id)
        self.connections[new_conn1.key] = new_conn1
        self.connections[new_conn2.key] = new_conn2
        return ng, conn_to_split  # the return is only used in genome_feedforward

    def mutate_add_connection(self, config):
        '''
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input nodes.
        '''
        possible_outputs = list(self.hidden.keys()) + list(self.outputs.keys())
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + list(self.inputs.keys())
        in_node = choice(possible_inputs)

        # Only create the connection if it doesn't already exist.
        key = (in_node, out_node)
        if key not in self.connections:
            # TODO: factor out new connection creation based on config
            weight = gauss(0, config.weight_stdev)
            enabled = choice([False, True])
            cg = ConnectionGene(in_node, out_node, weight, enabled)
            self.connections[cg.key] = cg

    def mutate_delete_node(self):
        # Do nothing if there are no hidden nodes.
        if not self.hidden:
            return -1

        del_key, del_node = choice(list(self.hidden.items()))

        connections_to_delete = set()
        for k, v in self.connections.items():
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.hidden[del_key]

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

        # Compute node gene differences.
        excess1 = 0
        excess2 = 0
        bias_diff = 0.0
        response_diff = 0.0
        activation_diff = 0
        num_common = 0
        node_gene_count1 = 0
        node_gene_count2 = 0
        for set_name in ('inputs', 'hidden', 'outputs'):
            node_genes1 = getattr(genome1, set_name)
            node_genes2 = getattr(genome2, set_name)

            node_gene_count1 += len(node_genes1)
            node_gene_count2 += len(node_genes2)

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

        return distance

    def size(self):
        '''Returns genome 'complexity', taken to be (number of hidden nodes, number of enabled connections)'''
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled is True])
        return len(self.hidden), num_enabled_connections

    def __str__(self):
        s = "Input Nodes:"
        for k, ng in iteritems(self.inputs):
            s += "{0} {1!s}\n\t".format(k, ng)
        s = "Output Nodes:"
        for k, ng in iteritems(self.outputs):
            s += "{0} {1!s}\n\t".format(k, ng)
        s = "Hidden Nodes:"
        for k, ng in iteritems(self.hidden):
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
            act_func = choice(config.activation_functions)
            node_gene = config.node_gene_type(activation_type=act_func)
            assert node_id not in self.hidden
            self.hidden[node_id] = node_gene
            node_id += 1

    @classmethod
    def create(cls, config, key):
        g = config.genome_type.create_unconnected(config, key)

        # Add hidden nodes if requested.
        if config.hidden_nodes > 0:
            g.add_hidden_nodes(config.hidden_nodes)

        # Add connections based on initial connectivity type.
        if config.initial_connection == 'fs_neat':
            g.connect_fs_neat()
        elif config.initial_connection == 'fully_connected':
            g.connect_full()
        elif config.initial_connection == 'partial':
            g.connect_partial(config)

        return g

    @staticmethod
    def create_node(config, node_id):
        return NodeGene(node_id, config.new_bias(), config.new_response(),
                        config.new_aggregation(), config.new_activation())

    # TODO: Can this be changed to not need a configuration object?
    @classmethod
    def create_unconnected(cls, config, key):
        '''Create a genome for a network with no hidden nodes and no connections.'''
        c = cls(key)
        node_id = 0
        # Create input node genes.
        for i in range(config.input_nodes):
            assert node_id not in c.inputs
            c.inputs[node_id] = cls.create_node(config, node_id)
            node_id += 1

        # Create output node genes.
        for i in range(config.output_nodes):
            act_func = choice(config.activation_functions)
            assert node_id not in c.outputs
            c.outputs[node_id] = cls.create_node(config, node_id)
            node_id += 1

        assert node_id == len(c.inputs) + len(c.outputs)
        return c

    def connect_fs_neat(self, config):
        """ Randomly connect one input to all hidden and output nodes (FS-NEAT). """
        # TODO: Factor out the gene creation.
        input_id = choice(self.inputs.keys())
        for output_id in list(self.hidden.keys()) + list(self.outputs.keys()):
            weight = gauss(0, config.weight_stdev)
            cg = config.conn_gene_type(input_id, output_id, weight, True)
            self.connections[cg.key] = cg

    def compute_full_connections(self):
        """ Create a fully-connected genome. """
        # Connect each input node to all hidden and output nodes.
        connections = []
        for input_id in self.inputs.keys():
            for output_id in list(self.hidden.keys()) + list(self.outputs.keys()):
                connections.append((input_id, output_id))

        # Connect each hidden node to all output nodes.
        for hidden_id in self.hidden.keys():
            for output_id in self.outputs.keys():
                connections.append((hidden_id, output_id))

        return connections

    def connect_full(self, config):
        """ Create a fully-connected genome. """
        # TODO: Factor out the gene creation.
        for input_id, output_id in self.compute_full_connections():
            weight = gauss(0, config.weight_stdev)
            cg = config.conn_gene_type(input_id, output_id, weight, True)
            self.connections[cg.key] = cg

    def connect_partial(self, config):
        # TODO: Factor out the gene creation.
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections()
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            weight = gauss(0, config.weight_stdev)
            cg = ConnectionGene(input_id, output_id, weight, True)
            self.connections[cg.key] = cg



# TODO: This class only differs from DefaultGenome in mutation behavior and
# the node_order member.  Its complexity suggests the bar is set too high
# for creating user-defined genome types.

class FFGenome(DefaultGenome):
    """ A genome for feed-forward neural networks. Feed-forward
        topologies are a particular case of Recurrent NNs.
    """

    def __init__(self, key):
        super(FFGenome, self).__init__(key)
        self.node_order = []  # hidden node order

    def inherit_genes(self, parent1, parent2):
        super(FFGenome, self).inherit_genes(parent1, parent2)

        self.node_order = list(parent1.node_order)

        assert len(self.node_order) == len(self.hidden)

    def mutate_add_node(self, config):
        # TODO: This method is overcomplicated, we should factor out the base class
        # capability, pick a valid place to insert, and then tell it to do that.
        result = super(FFGenome, self).mutate_add_node(config)
        ng, split_conn = result
        if ng is not None:
            # Add node to node order list: after the presynaptic node of the split connection
            # and before the postsynaptic node of the split connection
            if split_conn.input in self.hidden:
                mini = self.node_order.index(split_conn.input) + 1
            else:
                # Presynaptic node is an input node, not hidden node
                mini = 0
            if split_conn.output in self.hidden:
                maxi = self.node_order.index(split_conn.output)
            else:
                # Postsynaptic node is an output node, not hidden node
                maxi = len(self.node_order)
            self.node_order.insert(randint(mini, maxi), ng.key)
            assert len(self.node_order) == len(self.hidden)

        return ng, split_conn

    def mutate_add_connection(self, config):
        '''
        Attempt to add a new connection, with the restrictions that (1) the output node
        cannot be one of the network input nodes, and (2) the connection must be feed-forward.
        '''
        in_node = choice(list(self.inputs.keys()))
        out_node = choice(list(self.outputs.keys()))

        # Only create the connection if it's feed-forward and it doesn't already exist.
        if self.__is_connection_feedforward(in_node, out_node):
            key = (in_node, out_node)
            if key not in self.connections:
                weight = gauss(0, config.weight_stdev)
                enabled = choice([False, True])
                cg = ConnectionGene(in_node, out_node, weight, enabled)
                self.connections[cg.key] = cg

    def mutate_delete_node(self):
        deleted_id = super(FFGenome, self).mutate_delete_node()
        if deleted_id != -1:
            self.node_order.remove(deleted_id)

        #assert len(self.node_genes) >= self.num_inputs + self.num_outputs

    def __is_connection_feedforward(self, in_node, out_node):
        if in_node in self.inputs or out_node in self.outputs:
            return True

        assert in_node in self.node_order
        assert out_node in self.node_order
        return self.node_order.index(in_node) < self.node_order.index(out_node)

    def add_hidden_nodes(self, num_hidden, config):
        node_id = self.get_new_hidden_id()
        for i in range(num_hidden):
            act_func = choice(config.activation_functions)
            node_gene = config.node_gene_type()
            assert node_id not in self.hidden
            self.hidden[node_id] = node_gene
            self.node_order.append(node_gene.ID)
            node_id += 1

    def __str__(self):
        s = super(FFGenome, self).__str__()
        s += '\nNode order: ' + str(self.node_order)
        return s
