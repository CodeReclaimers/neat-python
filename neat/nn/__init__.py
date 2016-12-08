from neat.graphs import required_for_output, feed_forward_layers
from neat.six_util import itervalues, iteritems


class FeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise Exception("Expected {0} inputs, got {1}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            #print(node, func, bias, response, links)
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)
            #print("  v[{}] = {}({} + {} * {} = {}) = {}".format(node, act_func, bias, response, s, bias + response * s, self.values[node]))
        #print(self.values)

        return [self.values[i] for i in self.output_nodes]


def create_feed_forward_phenotype(genome, config):
    """ Receives a genome and returns its phenotype (a neural network). """

    # Gather expressed connections.
    connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]

    layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
    #print(layers)
    node_evals = []
    for layer in layers:
        for node in layer:
            inputs = []
            node_expr = []
            # TODO: This could be more efficient.
            for cg in itervalues(genome.connections):
                input, output = cg.key
                if output == node and cg.enabled:
                    inputs.append((input, cg.weight))
                    node_expr.append("v[%d] * %f" % (input, cg.weight))

            ng = genome.nodes[node]
            aggregation_function = config.genome_config.aggregation_function_defs[ng.aggregation]
            activation_function = config.genome_config.activation_defs.get(ng.activation)
            node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

            #print("  v[%d] = %s(%f + %f * %s(%s))" % (node, ng.activation, ng.bias, ng.response, ng.aggregation, ", ".join(node_expr)))

    return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)


class RecurrentNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        self.values = [{}, {}]
        for v in self.values:
            for k in inputs + outputs:
                v[k] = 0.0

            for node, activation, aggregation, bias, response, links in self.node_evals:
                v[node] = 0.0
                for i, w in links:
                    v[i] = 0.0
        self.active = 0

    def reset(self):
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise Exception("Expected {0} inputs, got {1}".format(len(self.input_nodes), len(inputs)))

        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, activation, aggregation, bias, response, links in self.node_evals:
            node_inputs = [ivalues[i] * w for i, w in links]
            s = aggregation(node_inputs)
            ovalues[node] = activation(bias + response * s)

        return [ovalues[i] for i in self.output_nodes]


def create_recurrent_phenotype(genome, config):
    """ Receives a genome and returns its phenotype (a recurrent neural network). """
    genome_config = config.genome_config
    required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections)

    # Gather inputs and expressed connections.
    node_inputs = {}
    for cg in itervalues(genome.connections):
        if not cg.enabled:
            continue

        i, o = cg.key
        if o not in required and i not in required:
            continue

        if o not in node_inputs:
            node_inputs[o] = [(i, cg.weight)]
        else:
            node_inputs[o].append((i, cg.weight))

    node_evals = []
    for node_key, inputs in iteritems(node_inputs):
        node = genome.nodes[node_key]
        activation_function = genome_config.activation_defs.get(node.activation)
        aggregation_function = genome_config.aggregation_function_defs.get(node.aggregation)
        node_evals.append((node_key, activation_function, aggregation_function, node.bias, node.response, inputs))

    return RecurrentNetwork(genome_config.input_keys, genome_config.output_keys, node_evals)
