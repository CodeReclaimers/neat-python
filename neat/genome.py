from math import fabs
from random import choice, gauss, randint, random, shuffle
import sys

# Instead of adding six as a dependency, this code was copied from the six
# implementation, six is Copyright (c) 2010-2015 Benjamin Peterson
if sys.version_info[0] == 3:
    def itervalues(d, **kw):
        return iter(d.values(**kw))

    def iteritems(d, **kw):
        return iter(d.items(**kw))
else:
    def itervalues(d, **kw):
        return iter(d.itervalues(**kw))

    def iteritems(d, **kw):
        return iter(d.iteritems(**kw))


class Genome(object):
    """ A genome for general recurrent neural networks. """

    def __init__(self, ID, config, parent1_id, parent2_id):
        self.ID = ID
        self.config = config
        self.num_inputs = config.input_nodes
        self.num_outputs = config.output_nodes

        # (id, gene) pairs for connection and node gene sets.
        self.conn_genes = {}
        self.node_genes = {}

        self.fitness = None
        self.species_id = None

        # my parents id: helps in tracking genome's genealogy
        self.parent1_id = parent1_id
        self.parent2_id = parent2_id

    def mutate(self):
        """ Mutates this genome """

        # TODO: Make a configuration item to choose whether or not multiple mutations can happen at once.

        if random() < self.config.prob_add_node:
            self.mutate_add_node()

        if random() < self.config.prob_add_conn:
            self.mutate_add_connection()

        if random() < self.config.prob_delete_node:
            self.mutate_delete_node()

        if random() < self.config.prob_delete_conn:
            self.mutate_delete_connection()

        # Mutate connection genes (weights, enabled, etc.).
        for cg in self.conn_genes.values():
            cg.mutate(self.config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.node_genes.values():
            if ng.type != 'INPUT':
                ng.mutate(self.config)

        return self

    def crossover(self, other, child_id):
        """ Crosses over parents' genomes and returns a child. """

        # Parents must belong to the same species.
        assert self.species_id == other.species_id, 'Different parents species ID: {0} vs {1}'.format(self.species_id,
                                                                                                      other.species_id)

        # TODO: if they're of equal fitness, choose the shortest
        if self.fitness > other.fitness:
            parent1 = self
            parent2 = other
        else:
            parent1 = other
            parent2 = self

        # creates a new child
        child = self.__class__(child_id, self.config, self.ID, other.ID)

        child.inherit_genes(parent1, parent2)

        child.species_id = parent1.species_id

        return child

    def inherit_genes(self, parent1, parent2):
        """ Applies the crossover operator. """
        assert (parent1.fitness >= parent2.fitness)

        # Crossover connection genes
        for cg1 in parent1.conn_genes.values():
            try:
                cg2 = parent2.conn_genes[cg1.key]
            except KeyError:
                # Copy excess or disjoint genes from the fittest parent
                self.conn_genes[cg1.key] = cg1.copy()
            else:
                if cg2.is_same_innov(cg1):  # Always true for *global* INs
                    # Homologous gene found
                    new_gene = cg1.get_child(cg2)
                else:
                    new_gene = cg1.copy()
                self.conn_genes[new_gene.key] = new_gene

        # Crossover node genes
        for ng1_id, ng1 in parent1.node_genes.items():
            ng2 = parent2.node_genes.get(ng1_id)
            if ng2 is None:
                # copies extra genes from the fittest parent
                new_gene = ng1.copy()
            else:
                # matching node genes: randomly selects the neuron's bias and response
                new_gene = ng1.get_child(ng2)

            assert new_gene.ID not in self.node_genes
            self.node_genes[new_gene.ID] = new_gene

    def get_new_hidden_id(self):
        new_id = 0
        while new_id in self.node_genes:
            new_id += 1
        return new_id

    def mutate_add_node(self):
        if not self.conn_genes:
            return None

        # Choose a random connection to split
        conn_to_split = choice(list(self.conn_genes.values()))
        new_node_id = self.get_new_hidden_id()
        act_func = choice(self.config.activation_functions)
        ng = self.config.node_gene_type(new_node_id, 'HIDDEN', activation_type=act_func)
        assert ng.ID not in self.node_genes
        self.node_genes[ng.ID] = ng
        new_conn1, new_conn2 = conn_to_split.split(ng.ID)
        self.conn_genes[new_conn1.key] = new_conn1
        self.conn_genes[new_conn2.key] = new_conn2
        return ng, conn_to_split  # the return is only used in genome_feedforward

    def mutate_add_connection(self):
        '''
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input nodes.
        '''
        in_node = choice(list(self.node_genes.values()))

        # TODO: We do this filtering of input/output/hidden nodes a lot;
        # they should probably be separate collections.
        possible_outputs = [n for n in self.node_genes.values() if n.type != 'INPUT']
        out_node = choice(possible_outputs)

        # Only create the connection if it doesn't already exist.
        key = (in_node.ID, out_node.ID)
        if key not in self.conn_genes:
            weight = gauss(0, self.config.weight_stdev)
            enabled = choice([False, True])
            cg = self.config.conn_gene_type(in_node.ID, out_node.ID, weight, enabled)
            self.conn_genes[cg.key] = cg

    def mutate_delete_node(self):
        # Do nothing if there are no hidden nodes.
        if len(self.node_genes) <= self.num_inputs + self.num_outputs:
            return -1

        idx = None
        while 1:
            idx = choice(list(self.node_genes.keys()))
            if self.node_genes[idx].type == 'HIDDEN':
                break

        node = self.node_genes[idx]
        node_id = node.ID

        keys_to_delete = set()
        for key, value in self.conn_genes.items():
            if node_id in (value.in_node_id, value.out_node_id):
                keys_to_delete.add(key)

        # Do not allow deletion of all connection genes.
        if len(keys_to_delete) >= len(self.conn_genes):
            return -1

        for key in keys_to_delete:
            del self.conn_genes[key]

        del self.node_genes[idx]

        assert len(self.conn_genes) > 0
        assert len(self.node_genes) >= self.num_inputs + self.num_outputs

        return node_id

    def mutate_delete_connection(self):
        if len(self.conn_genes) > self.num_inputs + self.num_outputs:
            key = choice(list(self.conn_genes.keys()))
            del self.conn_genes[key]

            assert len(self.conn_genes) > 0
            assert len(self.node_genes) >= self.num_inputs + self.num_outputs

    # compatibility function
    def distance(self, other):
        """ Returns the distance between this genome and the other. """
        if len(self.conn_genes) > len(other.conn_genes):
            node_genes1 = self.node_genes
            conn_genes1 = self.conn_genes
            node_genes2 = other.node_genes
            conn_genes2 = other.conn_genes

        else:
            node_genes1 = other.node_genes
            conn_genes1 = other.conn_genes
            node_genes2 = self.node_genes
            conn_genes2 = self.conn_genes

        # Compute node gene differences.
        excess1 = 0
        excess2 = sum(1 for k2 in node_genes2 if k2 not in node_genes1)
        bias_diff = 0.0
        response_diff = 0.0
        activation_diff = 0
        num_common = 0
        for k1, g1 in iteritems(node_genes1):
            if k1 in node_genes2:
                num_common += 1
                g2 = node_genes2[k1]
                bias_diff += fabs(g1.bias - g2.bias)
                response_diff += fabs(g1.response - g2.response)
                if g1.activation_type != g2.activation_type:
                    activation_diff += 1
            else:
                excess1 += 1

        most_nodes = max(len(node_genes1), len(node_genes2))
        distance = (self.config.excess_coefficient * float(excess1 + excess2) / most_nodes
                    + self.config.excess_coefficient * float(activation_diff) / most_nodes
                    + self.config.weight_coefficient * (bias_diff + response_diff) / num_common)

        # Compute connection gene differences.
        if conn_genes1:
            N = len(conn_genes1)
            weight_diff = 0
            matching = 0
            disjoint = 0
            excess = 0

            max_cg_genome2 = None
            if conn_genes2:
                max_cg_genome2 = max(itervalues(conn_genes2))

            for k1, cg1 in iteritems(conn_genes1):
                if k1 in conn_genes2:
                    # Homologous genes
                    cg2 = conn_genes2[k1]
                    weight_diff += fabs(cg1.weight - cg2.weight)
                    matching += 1

                    if cg1.enabled != cg2.enabled:
                        weight_diff += 1.0
                else:
                    if max_cg_genome2 is not None and cg1 > max_cg_genome2:
                        excess += 1
                    else:
                        disjoint += 1

            disjoint += len(conn_genes2) - matching

            distance += self.config.excess_coefficient * float(excess) / N
            distance += self.config.disjoint_coefficient * float(disjoint) / N
            if matching > 0:
                distance += self.config.weight_coefficient * (weight_diff / matching)

        return distance

    def size(self):
        '''Returns genome 'complexity', taken to be (number of hidden nodes, number of enabled connections)'''
        num_hidden_nodes = len(self.node_genes) - self.num_inputs - self.num_outputs
        num_enabled_connections = sum([1 for cg in self.conn_genes.values() if cg.enabled is True])
        return num_hidden_nodes, num_enabled_connections

    def __lt__(self, other):
        '''Order genomes by fitness.'''
        return self.fitness < other.fitness

    def __str__(self):
        s = "Nodes:"
        for ng in self.node_genes.values():
            s += "\n\t" + str(ng)
        s += "\nConnections:"
        connections = list(self.conn_genes.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s

    def add_hidden_nodes(self, num_hidden):
        node_id = self.get_new_hidden_id()
        for i in range(num_hidden):
            act_func = choice(self.config.activation_functions)
            node_gene = self.config.node_gene_type(node_id,
                                                   node_type='HIDDEN',
                                                   activation_type=act_func)
            assert node_gene.ID not in self.node_genes
            self.node_genes[node_gene.ID] = node_gene
            node_id += 1

    # TODO: Can this be changed to not need a configuration object?
    @classmethod
    def create_unconnected(cls, ID, config):
        '''Create a genome for a network with no hidden nodes and no connections.'''
        c = cls(ID, config, None, None)
        node_id = 0
        # Create input node genes.
        for i in range(config.input_nodes):
            assert node_id not in c.node_genes
            c.node_genes[node_id] = config.node_gene_type(node_id, 'INPUT')
            node_id += 1

        # Create output node genes.
        for i in range(config.output_nodes):
            act_func = choice(config.activation_functions)
            node_gene = config.node_gene_type(node_id,
                                              node_type='OUTPUT',
                                              activation_type=act_func)
            assert node_gene.ID not in c.node_genes
            c.node_genes[node_gene.ID] = node_gene
            node_id += 1

        assert node_id == len(c.node_genes)
        return c

    def connect_fs_neat(self):
        """ Randomly connect one input to all hidden and output nodes (FS-NEAT). """
        in_genes = [g for g in self.node_genes.values() if g.type == 'INPUT']
        hid_genes = [g for g in self.node_genes.values() if g.type == 'HIDDEN']
        out_genes = [g for g in self.node_genes.values() if g.type == 'OUTPUT']

        ig = choice(in_genes)
        for og in hid_genes + out_genes:
            weight = gauss(0, self.config.weight_stdev)
            cg = self.config.conn_gene_type(ig.ID, og.ID, weight, True)
            self.conn_genes[cg.key] = cg

    def compute_full_connections(self):
        """ Create a fully-connected genome. """
        in_genes = [g for g in self.node_genes.values() if g.type == 'INPUT']
        hid_genes = [g for g in self.node_genes.values() if g.type == 'HIDDEN']
        out_genes = [g for g in self.node_genes.values() if g.type == 'OUTPUT']

        # Connect each input node to all hidden and output nodes.
        connections = []
        for ig in in_genes:
            for og in hid_genes + out_genes:
                connections.append((ig.ID, og.ID))

        # Connect each hidden node to all output nodes.
        for hg in hid_genes:
            for og in out_genes:
                connections.append((hg.ID, og.ID))

        return connections

    def connect_full(self):
        """ Create a fully-connected genome. """
        for input_id, output_id in self.compute_full_connections():
            weight = gauss(0, self.config.weight_stdev)
            cg = self.config.conn_gene_type(input_id, output_id, weight, True)
            self.conn_genes[cg.key] = cg

    def connect_partial(self, fraction):
        assert 0 <= fraction <= 1
        all_connections = self.compute_full_connections()
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            weight = gauss(0, self.config.weight_stdev)
            cg = self.config.conn_gene_type(input_id, output_id, weight, True)
            self.conn_genes[cg.key] = cg


class FFGenome(Genome):
    """ A genome for feed-forward neural networks. Feed-forward
        topologies are a particular case of Recurrent NNs.
    """

    def __init__(self, ID, config, parent1_id, parent2_id):
        super(FFGenome, self).__init__(ID, config, parent1_id, parent2_id)
        self.node_order = []  # hidden node order

    def inherit_genes(self, parent1, parent2):
        super(FFGenome, self).inherit_genes(parent1, parent2)

        self.node_order = list(parent1.node_order)

        assert (len(self.node_order) == len([n for n in self.node_genes.values() if n.type == 'HIDDEN']))

    def mutate_add_node(self):
        result = super(FFGenome, self).mutate_add_node()
        if result is None:
            return

        ng, split_conn = result
        # Add node to node order list: after the presynaptic node of the split connection
        # and before the postsynaptic node of the split connection
        if self.node_genes[split_conn.in_node_id].type == 'HIDDEN':
            mini = self.node_order.index(split_conn.in_node_id) + 1
        else:
            # Presynaptic node is an input node, not hidden node
            mini = 0
        if self.node_genes[split_conn.out_node_id].type == 'HIDDEN':
            maxi = self.node_order.index(split_conn.out_node_id)
        else:
            # Postsynaptic node is an output node, not hidden node
            maxi = len(self.node_order)
        self.node_order.insert(randint(mini, maxi), ng.ID)
        assert (len(self.node_order) == len([n for n in self.node_genes.values() if n.type == 'HIDDEN']))
        return ng, split_conn

    def mutate_add_connection(self):
        '''
        Attempt to add a new connection, with the restrictions that (1) the output node
        cannot be one of the network input nodes, and (2) the connection must be feed-forward.
        '''
        possible_inputs = [n for n in self.node_genes.values() if n.type != 'OUTPUT']
        possible_outputs = [n for n in self.node_genes.values() if n.type != 'INPUT']

        in_node = choice(possible_inputs)
        out_node = choice(possible_outputs)

        # Only create the connection if it's feed-forward and it doesn't already exist.
        if self.__is_connection_feedforward(in_node, out_node):
            key = (in_node.ID, out_node.ID)
            if key not in self.conn_genes:
                weight = gauss(0, self.config.weight_stdev)
                enabled = choice([False, True])
                cg = self.config.conn_gene_type(in_node.ID, out_node.ID, weight, enabled)
                self.conn_genes[cg.key] = cg

    def mutate_delete_node(self):
        deleted_id = super(FFGenome, self).mutate_delete_node()
        if deleted_id != -1:
            self.node_order.remove(deleted_id)

        assert len(self.node_genes) >= self.num_inputs + self.num_outputs

    def __is_connection_feedforward(self, in_node, out_node):
        if in_node.type == 'INPUT' or out_node.type == 'OUTPUT':
            return True

        assert in_node.ID in self.node_order
        assert out_node.ID in self.node_order
        return self.node_order.index(in_node.ID) < self.node_order.index(out_node.ID)

    def add_hidden_nodes(self, num_hidden):
        node_id = self.get_new_hidden_id()
        for i in range(num_hidden):
            act_func = choice(self.config.activation_functions)
            node_gene = self.config.node_gene_type(node_id,
                                                   node_type='HIDDEN',
                                                   activation_type=act_func)
            assert node_gene.ID not in self.node_genes
            self.node_genes[node_gene.ID] = node_gene
            self.node_order.append(node_gene.ID)
            node_id += 1

    def __str__(self):
        s = super(FFGenome, self).__str__()
        s += '\nNode order: ' + str(self.node_order)
        return s
