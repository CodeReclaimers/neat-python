import random


class NodeGene(object):
    def __init__(self, ID, node_type, bias=0.0, response=4.924273, activation_type='sigmoid'):
        """ A node gene encodes the basic artificial neuron model.
            node_type must be 'INPUT', 'HIDDEN', or 'OUTPUT'
        """
        assert activation_type is not None
        # TODO: Node genes probably shouldn't need to know whether they are input/output/hidden.
        assert node_type in ('INPUT', 'OUTPUT', 'HIDDEN')

        self.ID = ID
        self.type = node_type
        self.bias = bias
        self.response = response
        self.activation_type = activation_type

    def __str__(self):
        return 'NodeGene(id={0}, type={1}, bias={2}, response={3}, activation={4})'.format(
            self.ID, self.type, self.bias, self.response, self.activation_type)

    def get_child(self, other):
        """ Creates a new NodeGene randomly inheriting attributes from its parents."""
        assert (self.ID == other.ID)

        ng = NodeGene(self.ID, self.type,
                      random.choice((self.bias, other.bias)),
                      random.choice((self.response, other.response)),
                      random.choice((self.activation_type, other.activation_type)))
        return ng

    def mutate_bias(self, config):
        new_bias = self.bias + random.gauss(0, 1) * config.bias_mutation_power
        self.bias = max(config.min_weight, min(config.max_weight, new_bias))

    def mutate_response(self, config):
        """ Mutates the neuron's average firing response. """
        new_response = self.response + random.gauss(0, 1) * config.response_mutation_power
        self.response = max(config.min_weight, min(config.max_weight, new_response))

    def mutate_activation(self, config):
        self.activation_type = random.choice(config.activation_functions)

    def copy(self):
        return NodeGene(self.ID, self.type, self.bias,
                        self.response, self.activation_type)

    def mutate(self, config):
        if random.random() < config.prob_mutate_bias:
            self.mutate_bias(config)
        if random.random() < config.prob_mutate_response:
            self.mutate_response(config)
        if random.random() < config.prob_mutate_activation:
            self.mutate_activation(config)


class ConnectionGene(object):
    def __init__(self, innovation_id, in_node_id, out_node_id, weight, enabled):
        assert type(innovation_id) is int
        assert type(in_node_id) is int
        assert type(out_node_id) is int
        assert type(weight) is float
        assert type(enabled) is bool

        self.innovation_id = innovation_id
        self.in_node_id = in_node_id
        self.out_node_id = out_node_id
        self.weight = weight
        self.enabled = enabled

    # Key for dictionaries, avoids two connections between the same nodes.
    key = property(lambda self: (self.in_node_id, self.out_node_id))

    def mutate(self, config):
        r = random.random
        if r() < config.prob_mutate_weight:
            if r() < config.prob_replace_weight:
                # Replace weight with a random value.
                self.weight = random.gauss(0, config.weight_stdev)
            else:
                # Perturb weight.
                new_weight = self.weight + random.gauss(0, 1) * config.weight_mutation_power
                self.weight = max(config.min_weight, min(config.max_weight, new_weight))

        if r() < config.prob_toggle_link:
            self.enabled = not self.enabled

    def enable(self):
        """ Enables a link. """
        self.enabled = True

    def __str__(self):
        return 'ConnectionGene(in={0}, out={1}, weight={2}, enabled={3}, innov={4})'.format(
            self.in_node_id, self.out_node_id, self.weight, self.enabled, self.innovation_id)

    def __lt__(self, other):
        return self.innovation_id < other.innovation_id

    def split(self, innovation_indexer, node_id):
        """ Splits a connection, creating two new connections and disabling this one """
        self.enabled = False

        innovation1 = innovation_indexer.get_innovation_id(self.in_node_id, node_id)
        new_conn1 = ConnectionGene(innovation1, self.in_node_id, node_id, 1.0, True)

        innovation2 = innovation_indexer.get_innovation_id(node_id, self.out_node_id)
        new_conn2 = ConnectionGene(innovation2, node_id, self.out_node_id, self.weight, True)

        return new_conn1, new_conn2

    def copy(self):
        return ConnectionGene(self.innovation_id, self.in_node_id, self.out_node_id, self.weight, self.enabled)

    def is_same_innov(self, cg):
        return self.innovation_id == cg.innovation_id

    def get_child(self, cg):
        # TODO: average both weights (Stanley, p. 38)
        return random.choice((self, cg)).copy()
