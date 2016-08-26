import copy

from neat import activation_functions
from neat.six_util import iterkeys, itervalues


def find_feed_forward_layers(inputs, connections):
    '''
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param connections: list of (input, output) connections in the network.

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    '''

    # TODO: Detect and omit nodes whose output is ultimately never used.

    layers = []
    S = set(inputs)
    while 1:
        # Find candidate nodes C for the next layer.  These nodes should connect
        # a node in S to a node not in S.
        C = set(b for (a, b) in connections if a in S and b not in S)
        # Keep only the nodes whose entire input set is contained in S.
        T = set()
        for n in C:
            if all(a in S for (a, b) in connections if b == n):
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
        self.values = [0.0] * (1 + max_node)

    def serial_activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise Exception("Expected {0} inputs, got {1}".format(len(self.input_nodes), len(inputs)))

        for i, v in zip(self.input_nodes, inputs):
            self.values[i] = v

        for node, func, bias, response, links in self.node_evals:
            s = 0.0
            for i, w in links:
                s += self.values[i] * w
            self.values[node] = func(bias + response * s)

        return [self.values[i] for i in self.output_nodes]


def create_feed_forward_phenotype(genome):
    """ Receives a genome and returns its phenotype (a neural network). """

    # Gather inputs and expressed connections.
    input_nodes = list(iterkeys(genome.inputs))
    output_nodes = list(iterkeys(genome.outputs))
    connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]

    # TODO: It seems like this might be worth factoring out somewhere.
    all_nodes = copy.copy(genome.inputs)
    all_nodes.update(genome.outputs)
    all_nodes.update(genome.hidden)

    layers = find_feed_forward_layers(input_nodes, connections)
    node_evals = []
    #used_nodes = set(input_nodes + output_nodes)
    max_used_node = max(max(input_nodes), max(output_nodes))
    for layer in layers:
        for node in layer:
            inputs = []
            # TODO: This could be more efficient.
            for cg in itervalues(genome.connections):
                if cg.output == node and cg.enabled:
                    inputs.append((cg.input, cg.weight))
                    #used_nodes.add(cg.in_node_id)
                    max_used_node = max(max_used_node, cg.input)

            #used_nodes.add(node)
            max_used_node = max(max_used_node, node)
            ng = all_nodes[node]
            activation_function = activation_functions.get(ng.activation)
            node_evals.append((node, activation_function, ng.bias, ng.response, inputs))

    return FeedForwardNetwork(max_used_node, input_nodes, output_nodes, node_evals)


class RecurrentNetwork(object):
    def __init__(self, max_node, inputs, outputs, node_evals):
        self.max_node = max_node
        self.node_evals = node_evals
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.reset()

    def reset(self):
        self.values = [[0.0] * (1 + self.max_node),
                       [0.0] * (1 + self.max_node)]
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


def create_recurrent_phenotype(genome):
    """ Receives a genome and returns its phenotype (a recurrent neural network). """

    # Gather inputs and expressed connections.
    node_inputs = {}
    used_nodes = {}
    used_nodes.update(genome.inputs)
    used_nodes.update(genome.outputs)
    for cg in genome.connections.values():
        if not cg.enabled:
            continue

        if cg.output not in used_nodes:
            used_nodes[cg.output] = genome.hidden[cg.output]

        if cg.input not in used_nodes:
            used_nodes[cg.input] = genome.hidden[cg.input]

        if cg.output not in node_inputs:
            node_inputs[cg.output] = [(cg.input, cg.weight)]
        else:
            node_inputs[cg.output].append((cg.input, cg.weight))

    node_evals = []
    for onode, inputs in node_inputs.items():
        ng = used_nodes[onode]
        activation_function = activation_functions.get(ng.activation)
        node_evals.append((onode, activation_function, ng.bias, ng.response, inputs))

    input_nodes = list(genome.inputs.keys())
    input_nodes.sort()
    output_nodes = list(genome.outputs.keys())
    output_nodes.sort()

    return RecurrentNetwork(max(used_nodes), input_nodes, output_nodes, node_evals)
