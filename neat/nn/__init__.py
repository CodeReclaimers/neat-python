from neat.six_util import iterkeys, itervalues, iteritems

# TODO: All this directed graph logic should be in a core module.


def creates_cycle(connections, test):
    """
    Returns true if the addition of the "test" connection would create a cycle,
    assuming that no cycle already exists in the graph represented by "connections".
    """
    i, o = test
    if i == o:
        return True

    visited = set([o])
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False


def required_for_output(inputs, outputs, connections):
    '''
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a list of layers, with each layer consisting of a set of identifiers.
    '''

    required = set(outputs)
    S = set(outputs)
    while 1:
        # Find nodes not in S whose output is consumed by a node in S.
        T = set(a for (a, b) in connections if b in S and a not in S)

        if not T:
            break

        layer_nodes = set(x for x in T if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        S = S.union(T)

    return required


def feed_forward_layers(inputs, outputs, connections):
    '''
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    '''

    required = required_for_output(inputs, outputs, connections)

    layers = []
    S = set(inputs)
    while 1:
        # Find candidate nodes C for the next layer.  These nodes should connect
        # a node in S to a node not in S.
        C = set(b for (a, b) in connections if a in S and b not in S)
        # Keep only the used nodes whose entire input set is contained in S.
        T = set()
        for n in C:
            if n in required and all(a in S for (a, b) in connections if b == n):
                T.add(n)

        if not T:
            break

        layers.append(T)
        S = S.union(T)

    return layers


class FeedForwardNetwork(object):
    def __init__(self, max_node, inputs, outputs, node_evals):
        self.node_evals = node_evals
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def serial_activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise Exception("Expected {0} inputs, got {1}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, agg_func, act_func, bias, response, links in self.node_evals:
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
    max_used_node = max(max(config.genome_config.input_keys), max(config.genome_config.output_keys))
    for layer in layers:
        for node in layer:
            inputs = []
            node_expr = []
            # TODO: This could be more efficient.
            for cg in itervalues(genome.connections):
                if cg.output == node and cg.enabled:
                    inputs.append((cg.input, cg.weight))
                    node_expr.append("v[%d] * %f" % (cg.input, cg.weight))
                    max_used_node = max(max_used_node, cg.input)

            max_used_node = max(max_used_node, node)
            ng = genome.nodes[node]
            aggregation_function = config.genome_config.aggregation_function_defs[ng.aggregation]
            activation_function = config.genome_config.activation_defs.get(ng.activation)
            node_evals.append((node, aggregation_function, activation_function, ng.bias, ng.response, inputs))

            #print("  v[%d] = %s(%f + %f * %s(%s))" % (node, ng.activation, ng.bias, ng.response, ng.aggregation, ", ".join(node_expr)))

    return FeedForwardNetwork(max_used_node, config.genome_config.input_keys, config.genome_config.output_keys, node_evals)


class RecurrentNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        self.values = [{}, {}]
        for v in self.values:
            for k in inputs + outputs:
                v[k] = 0.0

            for node, func, bias, response, links in self.node_evals:
                v[node] = 0.0
                for i, w in links:
                    v[i] = 0.0

    def reset(self):
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    def activate(self, inputs):
        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, func, bias, response, links in self.node_evals:
            s = 0.0
            for i, w in links:
                s += ivalues[i] * w
            ovalues[node] = func(bias + response * s)

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
        node_evals.append((node_key, activation_function, node.bias, node.response, inputs))

    return RecurrentNetwork(genome_config.input_keys, genome_config.output_keys, node_evals)
