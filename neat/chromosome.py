import random
import math
from config import Config
import genome

# Temporary workaround - default settings
# node_gene_type = genome.NodeGene
conn_gene_type = genome.ConnectionGene


class Chromosome(object):
    """ A chromosome for general recurrent neural networks. """
    _id = 0

    def __init__(self, parent1_id, parent2_id, node_gene_type, conn_gene_type):

        self._id = self.__get_new_id()
        self._input_nodes = Config.input_nodes
        self._output_nodes = Config.output_nodes

        # the type of NodeGene and ConnectionGene the chromosome carries
        self._node_gene_type = node_gene_type
        self._conn_gene_type = conn_gene_type
        # how many genes of the previous type the chromosome has
        self._connection_genes = {}  # dictionary of connection genes
        self._node_genes = []

        self.fitness = None
        self.species_id = None

        # my parents id: helps in tracking chromosome's genealogy
        self.parent1_id = parent1_id
        self.parent2_id = parent2_id

    conn_genes = property(lambda self: self._connection_genes.values())
    node_genes = property(lambda self: self._node_genes)
    sensors = property(lambda self: self._input_nodes)
    actuators = property(lambda self: self._output_nodes)
    id = property(lambda self: self._id)

    @classmethod
    def __get_new_id(cls):
        cls._id += 1
        return cls._id

    def mutate(self):
        """ Mutates this chromosome """

        r = random.random
        if r() < Config.prob_addnode:
            self._mutate_add_node()

        elif r() < Config.prob_addconn:
            self._mutate_add_connection()

        else:
            # mutate weights
            for cg in self._connection_genes.values():
                cg.mutate()

            # mutate bias, response, and etc...
            for ng in self._node_genes[self._input_nodes:]:
                ng.mutate()

        return self

    def crossover(self, other):
        """ Crosses over parents' chromosomes and returns a child. """

        # This can't happen! Parents must belong to the same species.
        assert self.species_id == other.species_id, 'Different parents species ID: %d vs %d' \
                                                    % (self.species_id, other.species_id)

        # TODO: if they're of equal fitnesses, choose the shortest
        if self.fitness > other.fitness:
            parent1 = self
            parent2 = other
        else:
            parent1 = other
            parent2 = self

        # creates a new child
        child = self.__class__(self.id, other.id, self._node_gene_type, self._conn_gene_type)

        child._inherit_genes(parent1, parent2)

        child.species_id = parent1.species_id
        # child._input_nodes = parent1._input_nodes

        return child

    def _inherit_genes(self, parent1, parent2):
        """ Applies the crossover operator. """
        assert (parent1.fitness >= parent2.fitness)

        # Crossover connection genes
        for cg1 in parent1._connection_genes.values():
            try:
                cg2 = parent2._connection_genes[cg1.key]
            except KeyError:
                # Copy excess or disjoint genes from the fittest parent
                self._connection_genes[cg1.key] = cg1.copy()
            else:
                if cg2.is_same_innov(cg1):  # Always true for *global* INs
                    # Homologous gene found
                    new_gene = cg1.get_child(cg2)
                    # new_gene.enable() # avoids disconnected neurons
                else:
                    new_gene = cg1.copy()
                self._connection_genes[new_gene.key] = new_gene

        # Crossover node genes
        for i, ng1 in enumerate(parent1._node_genes):
            try:
                # matching node genes: randomly selects the neuron's bias and response
                self._node_genes.append(ng1.get_child(parent2._node_genes[i]))
            except IndexError:
                # copies extra genes from the fittest parent
                self._node_genes.append(ng1.copy())

    def _mutate_add_node(self):
        # Choose a random connection to split
        conn_to_split = random.choice(self._connection_genes.values())
        ng = self._node_gene_type(len(self._node_genes) + 1, 'HIDDEN', activation_type=Config.nn_activation)
        self._node_genes.append(ng)
        new_conn1, new_conn2 = conn_to_split.split(ng.id)
        self._connection_genes[new_conn1.key] = new_conn1
        self._connection_genes[new_conn2.key] = new_conn2
        return (ng, conn_to_split)  # the return is only used in genome_feedforward

    def _mutate_add_connection(self):
        # Only for recurrent networks
        total_possible_conns = (len(self._node_genes) - self._input_nodes) \
                               * len(self._node_genes)
        remaining_conns = total_possible_conns - len(self._connection_genes)
        # Check if new connection can be added:
        if remaining_conns > 0:
            n = random.randint(0, remaining_conns - 1)
            count = 0
            # Count connections
            for in_node in self._node_genes:
                for out_node in self._node_genes[self._input_nodes:]:
                    if (in_node.id, out_node.id) not in self._connection_genes.keys():
                        # Free connection
                        if count == n:  # Connection to create
                            weight = random.gauss(0, Config.weight_stdev)
                            cg = self._conn_gene_type(in_node.id, out_node.id, weight, True)
                            self._connection_genes[cg.key] = cg
                            return
                        else:
                            count += 1

    # compatibility function
    def distance(self, other):
        """ Returns the distance between this chromosome and the other. """
        if len(self._connection_genes) > len(other._connection_genes):
            chromo1 = self
            chromo2 = other
        else:
            chromo1 = other
            chromo2 = self

        weight_diff = 0
        matching = 0
        disjoint = 0
        excess = 0

        max_cg_chromo2 = max(chromo2._connection_genes.values())

        for cg1 in chromo1._connection_genes.values():
            try:
                cg2 = chromo2._connection_genes[cg1.key]
            except KeyError:
                if cg1 > max_cg_chromo2:
                    excess += 1
                else:
                    disjoint += 1
            else:
                # Homologous genes
                weight_diff += math.fabs(cg1.weight - cg2.weight)
                matching += 1

        disjoint += len(chromo2._connection_genes) - matching

        # assert(matching > 0) # this can't happen
        distance = Config.excess_coefficient * excess + Config.disjoint_coefficient * disjoint
        if matching > 0:
            distance += Config.weight_coefficient * (weight_diff / matching)

        return distance

    def size(self):
        """ Defines chromosome 'complexity': number of hidden nodes plus
            number of enabled connections (bias is not considered)
        """
        # number of hidden nodes
        num_hidden = len(self._node_genes) - self._input_nodes - self._output_nodes
        # number of enabled connections
        conns_enabled = sum([1 for cg in self._connection_genes.values() if cg.enabled is True])

        return (num_hidden, conns_enabled)

    def __cmp__(self, other):
        """ First compare chromosomes by their fitness and then by their id.
            Older chromosomes (lower ids) should be preferred if newer ones
            performs the same.
        """
        # return cmp(self.fitness, other.fitness) or cmp(other.id, self.id)
        return cmp(self.fitness, other.fitness)

    def __str__(self):
        s = "Nodes:"
        for ng in self._node_genes:
            s += "\n\t" + str(ng)
        s += "\nConnections:"
        connections = self._connection_genes.values()
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s

    def add_hidden_nodes(self, num_hidden):
        id = len(self._node_genes) + 1
        for i in range(num_hidden):
            node_gene = self._node_gene_type(id,
                                             nodetype='HIDDEN',
                                             activation_type=Config.nn_activation)
            self._node_genes.append(node_gene)
            id += 1
            # Connect all nodes to it
            for pre in self._node_genes:
                weight = random.gauss(0, Config.weight_stdev)
                cg = self._conn_gene_type(pre.id, node_gene.id, weight, True)
                self._connection_genes[cg.key] = cg
            # Connect it to all nodes except input nodes
            for post in self._node_genes[self._input_nodes:]:
                weight = random.gauss(0, Config.weight_stdev)
                cg = self._conn_gene_type(node_gene.id, post.id, weight, True)
                self._connection_genes[cg.key] = cg

    @classmethod
    def create_unconnected(cls):
        """
        Factory method
        Creates a chromosome for an unconnected feedforward network with no hidden nodes.
        """
        c = cls(0, 0, node_gene_type, conn_gene_type)
        id = 1
        # Create node genes
        for i in range(Config.input_nodes):
            c._node_genes.append(c._node_gene_type(id, 'INPUT'))
            id += 1
        # c._input_nodes += num_input
        for i in range(Config.output_nodes):
            node_gene = c._node_gene_type(id,
                                          nodetype='OUTPUT',
                                          activation_type=Config.nn_activation)
            c._node_genes.append(node_gene)
            id += 1
        assert id == len(c._node_genes) + 1
        return c

    @classmethod
    def create_minimally_connected(cls):
        """
        Factory method
        Creates a chromosome for a minimally connected feedforward network with no hidden nodes. That is,
        each output node will have a single connection from a randomly chosen input node.
        """
        c = cls.create_unconnected()
        for node_gene in c.node_genes:
            if node_gene.type != 'OUTPUT':
                continue

            # Connect it to a random input node
            input_node = random.choice(c._node_genes[:Config.input_nodes])
            weight = random.gauss(0, Config.weight_stdev)

            cg = c._conn_gene_type(input_node.id, node_gene.id, weight, True)
            c._connection_genes[cg.key] = cg

        return c

    @classmethod
    def create_fully_connected(cls):
        """
        Factory method
        Creates a chromosome for a fully connected feedforward network with no hidden nodes.
        """
        c = cls.create_unconnected()
        for node_gene in c.node_genes:
            if node_gene.type != 'OUTPUT':
                continue

            # Connect it to all input nodes
            for input_node in c._node_genes[:Config.input_nodes]:
                # TODO: review the initial weights distribution
                # weight = random.uniform(-1, 1)*Config.random_range
                weight = random.gauss(0, Config.weight_stdev)

                cg = c._conn_gene_type(input_node.id, node_gene.id, weight, True)
                c._connection_genes[cg.key] = cg

        return c


class FFChromosome(Chromosome):
    """ A chromosome for feedforward neural networks. Feedforward
        topologies are a particular case of Recurrent NNs.
    """

    def __init__(self, parent1_id, parent2_id, node_gene_type, conn_gene_type):
        super(FFChromosome, self).__init__(parent1_id, parent2_id, node_gene_type, conn_gene_type)
        self.__node_order = []  # hidden node order (for feedforward networks)

    node_order = property(lambda self: self.__node_order)

    def _inherit_genes(self, parent1, parent2):
        super(FFChromosome, self)._inherit_genes(parent1, parent2)

        self.__node_order = parent1.__node_order[:]

        assert (len(self.__node_order) == len([n for n in self.node_genes if n.type == 'HIDDEN']))

    def _mutate_add_node(self):
        ng, split_conn = super(FFChromosome, self)._mutate_add_node()
        # Add node to node order list: after the presynaptic node of the split connection
        # and before the postsynaptic node of the split connection
        if self._node_genes[split_conn.innodeid - 1].type == 'HIDDEN':
            mini = self.__node_order.index(split_conn.innodeid) + 1
        else:
            # Presynaptic node is an input node, not hidden node
            mini = 0
        if self._node_genes[split_conn.outnodeid - 1].type == 'HIDDEN':
            maxi = self.__node_order.index(split_conn.outnodeid)
        else:
            # Postsynaptic node is an output node, not hidden node
            maxi = len(self.__node_order)
        self.__node_order.insert(random.randint(mini, maxi), ng.id)
        assert (len(self.__node_order) == len([n for n in self.node_genes if n.type == 'HIDDEN']))
        return (ng, split_conn)

    def _mutate_add_connection(self):
        # Only for feedforwad networks
        num_hidden = len(self.__node_order)
        num_output = len(self._node_genes) - self._input_nodes - num_hidden

        total_possible_conns = (num_hidden + num_output) * (self._input_nodes + num_hidden) - \
                               sum(range(num_hidden + 1))

        remaining_conns = total_possible_conns - len(self._connection_genes)
        # Check if new connection can be added:
        if remaining_conns > 0:
            n = random.randint(0, remaining_conns - 1)
            count = 0
            # Count connections
            for in_node in (self._node_genes[:self._input_nodes] + self._node_genes[-num_hidden:]):
                for out_node in self._node_genes[self._input_nodes:]:
                    if (in_node.id, out_node.id) not in self._connection_genes.keys() and \
                            self.__is_connection_feedforward(in_node, out_node):
                        # Free connection
                        if count == n:  # Connection to create
                            # weight = random.uniform(-Config.random_range, Config.random_range)
                            weight = random.gauss(0, 1)
                            cg = self._conn_gene_type(in_node.id, out_node.id, weight, True)
                            self._connection_genes[cg.key] = cg
                            return
                        else:
                            count += 1

    def __is_connection_feedforward(self, in_node, out_node):
        return in_node.type == 'INPUT' or out_node.type == 'OUTPUT' or \
               self.__node_order.index(in_node.id) < self.__node_order.index(out_node.id)

    def add_hidden_nodes(self, num_hidden):
        id = len(self._node_genes) + 1
        for i in range(num_hidden):
            node_gene = self._node_gene_type(id,
                                             nodetype='HIDDEN',
                                             activation_type=Config.nn_activation)
            self._node_genes.append(node_gene)
            self.__node_order.append(node_gene.id)
            id += 1
            # Connect all input nodes to it
            for pre in self._node_genes[:self._input_nodes]:
                weight = random.gauss(0, Config.weight_stdev)
                cg = self._conn_gene_type(pre.id, node_gene.id, weight, True)
                self._connection_genes[cg.key] = cg
                assert self.__is_connection_feedforward(pre, node_gene)
            # Connect all previous hidden nodes to it
            for pre_id in self.__node_order[:-1]:
                assert pre_id != node_gene.id
                weight = random.gauss(0, Config.weight_stdev)
                cg = self._conn_gene_type(pre_id, node_gene.id, weight, True)
                self._connection_genes[cg.key] = cg
            # Connect it to all output nodes
            for post in self._node_genes[self._input_nodes:(self._input_nodes + self._output_nodes)]:
                assert post.type == 'OUTPUT'
                weight = random.gauss(0, Config.weight_stdev)
                cg = self._conn_gene_type(node_gene.id, post.id, weight, True)
                self._connection_genes[cg.key] = cg
                assert self.__is_connection_feedforward(node_gene, post)

    def __str__(self):
        s = super(FFChromosome, self).__str__()
        s += '\nNode order: ' + str(self.__node_order)
        return s


if __name__ == '__main__':
    # Example
    import visualize
    # define some attributes
    node_gene_type = genome.NodeGene  # standard neuron model
    conn_gene_type = genome.ConnectionGene  # and connection link
    Config.nn_activation = 'exp'  # activation function
    Config.weight_stdev = 0.9  # weights distribution

    Config.input_nodes = 2  # number of inputs
    Config.output_nodes = 1  # number of outputs

    # creates a chromosome for recurrent networks
    # c1 = Chromosome.create_fully_connected()

    # creates a chromosome for feedforward networks
    c2 = FFChromosome.create_fully_connected()
    # add two hidden nodes
    c2.add_hidden_nodes(2)
    # apply some mutations
    # c2._mutate_add_node()
    # c2._mutate_add_connection()

    # check the result
    # visualize.draw_net(c1) # for recurrent nets
    visualize.draw_net(c2)  # for feedforward nets
    # print the chromosome
    print c2
