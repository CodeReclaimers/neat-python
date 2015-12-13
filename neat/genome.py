from random import choice, gauss, randint, random
import math
from neat.indexer import Indexer


class Genome(object):
    """ A genome for general recurrent neural networks. """
    indexer = Indexer(1)

    def __init__(self, config, parent1_id, parent2_id, node_gene_type, conn_gene_type):
        self.config = config
        self.ID = Genome.indexer.next()
        self.num_inputs = config.input_nodes
        self.num_outputs = config.output_nodes

        # The types of node and connection genes the genome carries.
        self._node_gene_type = node_gene_type
        self._conn_gene_type = conn_gene_type

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
            self._mutate_add_node()

        if random() < self.config.prob_add_conn:
            self._mutate_add_connection()

        if random() < self.config.prob_delete_node:
            self._mutate_delete_node()

        if random() < self.config.prob_delete_conn:
            self._mutate_delete_connection()

        # Mutate connection genes (weights, enabled, etc.).
        for cg in self.conn_genes.values():
            cg.mutate(self.config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.node_genes.values():
            if ng.type != 'INPUT':
                ng.mutate(self.config)

        return self

    def crossover(self, other):
        """ Crosses over parents' genomes and returns a child. """

        # Parents must belong to the same species.
        assert self.species_id == other.species_id, 'Different parents species ID: {0} vs {1}'.format(self.species_id, other.species_id)

        # TODO: if they're of equal fitness, choose the shortest
        if self.fitness > other.fitness:
            parent1 = self
            parent2 = other
        else:
            parent1 = other
            parent2 = self

        # creates a new child
        child = self.__class__(self.config, self.ID, other.ID, self._node_gene_type, self._conn_gene_type)

        child._inherit_genes(parent1, parent2)

        child.species_id = parent1.species_id

        return child

    def _inherit_genes(self, parent1, parent2):
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
            ng2 = parent2.node_genes.get(ng1_id, None)
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

    def _mutate_add_node(self):
        # Choose a random connection to split
        conn_to_split = choice(list(self.conn_genes.values()))
        new_node_id = self.get_new_hidden_id()
        ng = self._node_gene_type(new_node_id, 'HIDDEN', activation_type=self.config.nn_activation)
        assert ng.ID not in self.node_genes
        self.node_genes[ng.ID] = ng
        new_conn1, new_conn2 = conn_to_split.split(ng.ID)
        self.conn_genes[new_conn1.key] = new_conn1
        self.conn_genes[new_conn2.key] = new_conn2
        return ng, conn_to_split  # the return is only used in genome_feedforward

    def _mutate_add_connection(self):
        # Only for recurrent networks
        total_possible_conns = (len(self.node_genes) - self.num_inputs) * len(self.node_genes)
        remaining_conns = total_possible_conns - len(self.conn_genes)
        # Check if new connection can be added:
        if remaining_conns > 0:
            n = randint(0, remaining_conns - 1)
            count = 0
            # Count connections
            for in_node in self.node_genes.values():
                for out_node in self.node_genes.values():
                    # TODO: We do this filtering of input/output/hidden nodes a lot; they should probably
                    # be separate collections.
                    if out_node.type == 'INPUT':
                        continue

                    if (in_node.ID, out_node.ID) not in self.conn_genes.keys():
                        # Free connection
                        if count == n:  # Connection to create
                            weight = gauss(0, self.config.weight_stdev)
                            cg = self._conn_gene_type(in_node.ID, out_node.ID, weight, True)
                            self.conn_genes[cg.key] = cg
                            return
                        else:
                            count += 1

    def _mutate_delete_node(self):
        # Do nothing if there are no hidden nodes.
        if len(self.node_genes) <= self.num_inputs + self.num_outputs:
            return -1

        while 1:
            idx = choice(list(self.node_genes.keys()))
            if self.node_genes[idx].type == 'HIDDEN':
                break

        node = self.node_genes[idx]
        node_id = node.ID

        keys_to_delete = set()
        for key, value in self.conn_genes.items():
            if value.in_node_id == node_id or value.out_node_id == node_id:
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

    def _mutate_delete_connection(self):
        if len(self.conn_genes) > self.num_inputs + self.num_outputs:
            key = choice(list(self.conn_genes.keys()))
            del self.conn_genes[key]

            assert len(self.conn_genes) > 0
            assert len(self.node_genes) >= self.num_inputs + self.num_outputs

    # compatibility function
    def distance(self, other):
        """ Returns the distance between this genome and the other. """
        if len(self.conn_genes) > len(other.conn_genes):
            genome1 = self
            genome2 = other
        else:
            genome1 = other
            genome2 = self

        N = len(genome1.conn_genes)
        weight_diff = 0
        matching = 0
        disjoint = 0
        excess = 0

        max_cg_genome2 = max(genome2.conn_genes.values())

        for cg1 in genome1.conn_genes.values():
            try:
                cg2 = genome2.conn_genes[cg1.key]
            except KeyError:
                if cg1 > max_cg_genome2:
                    excess += 1
                else:
                    disjoint += 1
            else:
                # Homologous genes
                weight_diff += math.fabs(cg1.weight - cg2.weight)
                matching += 1

        disjoint += len(genome2.conn_genes) - matching

        distance = self.config.excess_coefficient * float(excess) / N + self.config.disjoint_coefficient * float(disjoint) / N
        if matching > 0:
            distance += self.config.weight_coefficient * (weight_diff / matching)

        return distance

    def size(self):
        """ Defines genome 'complexity': number of hidden nodes plus
            number of enabled connections (bias is not considered)
        """
        # number of hidden nodes
        num_hidden = len(self.node_genes) - self.num_inputs - self.num_outputs
        # number of enabled connections
        conns_enabled = sum([1 for cg in self.conn_genes.values() if cg.enabled is True])

        return num_hidden, conns_enabled

    def __lt__(self, other):
        """
        Compare genomes by their fitness.
        """
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
            node_gene = self._node_gene_type(node_id,
                                             node_type='HIDDEN',
                                             activation_type=self.config.nn_activation)
            assert node_gene.ID not in self.node_genes
            self.node_genes[node_gene.ID] = node_gene
            node_id += 1
            # Connect all nodes to it
            for pre in self.node_genes.values():
                weight = gauss(0, self.config.weight_stdev)
                cg = self._conn_gene_type(pre.ID, node_gene.ID, weight, True)
                self.conn_genes[cg.key] = cg
            # Connect it to all nodes except input nodes
            for post in self.node_genes.values():
                if post.type == 'INPUT':
                    continue

                weight = gauss(0, self.config.weight_stdev)
                cg = self._conn_gene_type(node_gene.ID, post.ID, weight, True)
                self.conn_genes[cg.key] = cg

    @classmethod
    def create_unconnected(cls, config, node_gene_type, conn_gene_type):
        """
        Factory method
        Creates a genome for an unconnected feed-forward network with no hidden nodes.
        """
        c = cls(config, 0, 0, node_gene_type, conn_gene_type)
        node_id = 0
        # Create node genes
        for i in range(config.input_nodes):
            assert node_id not in c.node_genes
            c.node_genes[node_id] = c._node_gene_type(node_id, 'INPUT')
            node_id += 1
        # c.num_inputs += num_input
        for i in range(config.output_nodes):
            node_gene = c._node_gene_type(node_id,
                                          node_type='OUTPUT',
                                          activation_type=config.nn_activation)
            assert node_gene.ID not in c.node_genes
            c.node_genes[node_gene.ID] = node_gene
            node_id += 1
        assert node_id == len(c.node_genes)
        return c

    @classmethod
    def create_minimally_connected(cls, config, node_gene_type, conn_gene_type):
        """
        Factory method
        Creates a genome for a minimally connected feed-forward network with no hidden nodes. That is,
        each output node will have a single connection from a randomly chosen input node.
        """
        c = cls.create_unconnected(config, node_gene_type, conn_gene_type)
        for node_gene in c.node_genes.values():
            if node_gene.type != 'OUTPUT':
                continue

            # Connect it to a random input node
            while 1:
                idx = choice(list(c.node_genes.keys()))
                if c.node_genes[idx].type == 'INPUT':
                    break

            input_node = c.node_genes[idx]
            weight = gauss(0, config.weight_stdev)

            cg = c._conn_gene_type(input_node.ID, node_gene.ID, weight, True)
            c.conn_genes[cg.key] = cg

        return c

    @classmethod
    def create_fully_connected(cls, config, node_gene_type, conn_gene_type):
        """
        Factory method
        Creates a genome for a fully connected feed-forward network with no hidden nodes.
        """
        c = cls.create_unconnected(config, node_gene_type, conn_gene_type)
        for node_gene in c.node_genes.values():
            if node_gene.type != 'OUTPUT':
                continue

            # Connect it to all input nodes
            for input_node in c.node_genes.values():
                if input_node.type == 'INPUT':
                    weight = gauss(0, config.weight_stdev)
                    cg = c._conn_gene_type(input_node.ID, node_gene.ID, weight, True)
                    c.conn_genes[cg.key] = cg

        return c


class FFGenome(Genome):
    """ A genome for feed-forward neural networks. Feed-forward
        topologies are a particular case of Recurrent NNs.
    """

    def __init__(self, config, parent1_id, parent2_id, node_gene_type, conn_gene_type):
        super(FFGenome, self).__init__(config, parent1_id, parent2_id, node_gene_type, conn_gene_type)
        self.node_order = []  # hidden node order (for feed-forward networks)

    def _inherit_genes(self, parent1, parent2):
        super(FFGenome, self)._inherit_genes(parent1, parent2)

        self.node_order = list(parent1.node_order)

        assert (len(self.node_order) == len([n for n in self.node_genes.values() if n.type == 'HIDDEN']))

    def _mutate_add_node(self):
        ng, split_conn = super(FFGenome, self)._mutate_add_node()
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

    def _mutate_add_connection(self):
        # Only for feed-forward networks
        nhidden = len(self.node_order)
        nout = len(self.node_genes) - self.num_inputs - nhidden

        total_possible_conns = (nhidden + nout) * (self.num_inputs + nhidden) - sum(range(nhidden + 1))

        remaining_conns = total_possible_conns - len(self.conn_genes)
        # Check if new connection can be added:
        if remaining_conns > 0:
            n = randint(0, remaining_conns - 1)
            count = 0
            # Count connections
            for in_node in self.node_genes.values():
                if in_node.type == 'OUTPUT':
                    continue

                for out_node in self.node_genes.values():
                    if out_node.type == 'INPUT':
                        continue

                    if (in_node.ID, out_node.ID) not in self.conn_genes.keys() and \
                            self.__is_connection_feedforward(in_node, out_node):
                        # Free connection
                        if count == n:  # Connection to create
                            weight = gauss(0, self.config.weight_stdev)
                            cg = self._conn_gene_type(in_node.ID, out_node.ID, weight, True)
                            self.conn_genes[cg.key] = cg
                            return
                        else:
                            count += 1

    def _mutate_delete_node(self):
        deleted_id = super(FFGenome, self)._mutate_delete_node()
        if deleted_id != -1:
            self.node_order.remove(deleted_id)

        assert len(self.conn_genes) > 0
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
            node_gene = self._node_gene_type(node_id,
                                             node_type='HIDDEN',
                                             activation_type=self.config.nn_activation)
            assert node_gene.ID not in self.node_genes
            self.node_genes[node_gene.ID] = node_gene
            self.node_order.append(node_gene.ID)
            node_id += 1
            # Connect all input nodes to it
            for pre in self.node_genes.values():
                if pre.type == 'INPUT':
                    weight = gauss(0, self.config.weight_stdev)
                    cg = self._conn_gene_type(pre.ID, node_gene.ID, weight, True)
                    self.conn_genes[cg.key] = cg
                    assert self.__is_connection_feedforward(pre, node_gene)
            # Connect all previous hidden nodes to it
            for pre_id in self.node_order[:-1]:
                assert pre_id != node_gene.ID
                weight = gauss(0, self.config.weight_stdev)
                cg = self._conn_gene_type(pre_id, node_gene.ID, weight, True)
                self.conn_genes[cg.key] = cg
            # Connect it to all output nodes
            for post in self.node_genes.values():
                if post.type == 'OUTPUT':
                    weight = gauss(0, self.config.weight_stdev)
                    cg = self._conn_gene_type(node_gene.ID, post.ID, weight, True)
                    self.conn_genes[cg.key] = cg
                    assert self.__is_connection_feedforward(node_gene, post)

    def __str__(self):
        s = super(FFGenome, self).__str__()
        s += '\nNode order: ' + str(self.node_order)
        return s
