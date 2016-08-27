from random import random

# TODO: Evaluate using __slots__ for performance/memory usage improvement.


class NodeGene(object):
    """ Encodes parameters for a single artificial neuron. """

    def __init__(self, key, bias, response, aggregation, activation):
        # TODO: Move these asserts into an external validation mechanism that can be omitted at runtime if desired.
        # TODO: Validate aggregation and activation against current configuration.
        assert type(bias) is float
        assert type(response) is float
        assert type(aggregation) is str
        assert type(activation) is str

        self.key = key
        self.bias = bias
        self.response = response
        self.aggregation = aggregation
        self.activation = activation

    def __str__(self):
        return 'NodeGene(key= {0}, bias={1}, response={2}, aggregation={3}, activation={4})'.format(
            self.key, self.bias, self.response, self.aggregation, self.activation)

    def crossover(self, gene2):
        """ Creates a new NodeGene randomly inheriting attributes from its parents."""
        # TODO: Move these asserts into an external validation mechanism that can be omitted at runtime if desired.
        assert isinstance(self, NodeGene)
        assert isinstance(gene2, NodeGene)
        assert self.key == gene2.key

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        bias = self.bias if random() > 0.5 else gene2.bias
        response = self.response if random() > 0.5 else gene2.response
        aggregation = self.aggregation if random() > 0.5 else gene2.aggregation
        activation = self.activation if random() > 0.5 else gene2.activation
        ng = NodeGene(self.key, bias, response, aggregation, activation)
        return ng

    def copy(self):
        return NodeGene(self.key, self.bias, self.response, self.aggregation, self.activation)

    # TODO: Factor out mutation into a separate class.
    def mutate(self, config):
        self.bias = config.mutate_bias(self.bias)
        self.response = config.mutate_response(self.response)
        self.aggregation = config.mutate_aggregation(self.aggregation)
        self.activation = config.mutate_activation(self.activation)


# TODO: Evaluate using __slots__ for performance/memory usage improvement.

class ConnectionGene(object):
    def __init__(self, input_id, output_id, weight, enabled):
        # TODO: Move these asserts into an external validation mechanism that can be omitted at runtime if desired.
        assert type(input_id) is int
        assert type(output_id) is int
        assert type(weight) is float
        assert type(enabled) is bool

        self.key = (input_id, output_id)
        self.input = input_id
        self.output = output_id
        self.weight = weight
        # TODO: Do an ablation study to determine whether the enabled setting is
        # important--presumably mutations that set the weight to near zero could
        # provide a similar effect depending on the weight range and mutation rate.
        self.enabled = enabled

    # TODO: Factor out mutation into a separate class.
    def mutate(self, config):
        self.weight = config.mutate_weight(self.weight)

        if random() < config.prob_toggle_link:
            self.enabled = not self.enabled

    def __str__(self):
        return 'ConnectionGene(in={0}, out={1}, weight={2}, enabled={3}, innovation={4})'.format(
            self.input, self.output, self.weight, self.enabled, self.key)

    def __lt__(self, other):
        return self.key < other.key

    def split(self, node_id):
        """
        Disable this connection and create two new connections joining its nodes via
        the given node.  The new node+connections have roughly the same behavior as
        the original connection (depending on the activation function of the new node).
        """
        self.enabled = False
        new_conn1 = ConnectionGene(self.input, node_id, 1.0, True)
        new_conn2 = ConnectionGene(node_id, self.output, self.weight, True)

        return new_conn1, new_conn2

    def copy(self):
        return ConnectionGene(self.input, self.output, self.weight, self.enabled)

    def crossover(self, gene2):
        """ Creates a new ConnectionGene randomly inheriting attributes from its parents."""
        # TODO: Move these asserts into an external validation mechanism that can be omitted at runtime if desired.
        assert isinstance(self, ConnectionGene)
        assert isinstance(gene2, ConnectionGene)
        assert self.key == gene2.key

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        weight = self.weight if random() > 0.5 else gene2.weight
        enabled = self.enabled if random() > 0.5 else gene2.enabled
        return ConnectionGene(self.input, self.output, weight, enabled)
