from neat.genes import BaseGene, DefaultConnectionGene, DefaultNodeGene
from neat.genome import DefaultGenome, DefaultGenomeConfig
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute, DynamicAttribute
from neat.config import ConfigParameter
# from neat.graphs import required_for_output, feed_forward_layers
from neat.six_util import itervalues, iteritems
from random import choice, gauss, random, uniform
from copy import deepcopy


DEFAULT_READ_GATE = 1.0
DEFAULT_FORGET_GATE = 0.0


class GRUGenomeConfig(DefaultGenomeConfig):
    gru_params = [ConfigParameter('gru_prob', float)]

    def __init__(self, *args, **kwargs):
        DefaultGenomeConfig.__init__(self, *args, **kwargs)
        params = kwargs['params']
        for p in self.gru_params:
            self._params.append(p)
            setattr(self, p.name, p.interpret(params))


class GRUNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('bias'),
                        FloatAttribute('response'),
                        StringAttribute('activation', options='tanh'),
                        StringAttribute('aggregation', options='sum'),
                        DynamicAttribute('read_gate', options=[]),
                        DynamicAttribute('forget_gate')]
    _read_gate_idx = 4
    _forget_gate_idx = 5
    def __init__(self, key):
        assert isinstance(key, int), "GRUNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    # TODO: Gates should influence the distance
    def distance(self, other, config):
        d = abs(self.bias - other.bias) + abs(self.response - other.response)
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient

    def mutate(self, *args, **kwargs):
        # We need to update the gate options list before mutating
        pass

    def mutate_safe(self, *args, **kwargs):
        BaseGene.mutate(self, *args, **kwargs)

    def inform_deleted(self, key, config):
            if self.read_gate == key:
                self.read_gate = self._gene_attributes[self._read_gate_idx].init_value(config)
            if self.forget_gate == key:
                self.forget_gate = self._gene_attributes[self._forget_gate_idx].init_value(config)


class GRUGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = GRUNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        config = GRUGenomeConfig(params=param_dict)
        return config

    def configure_new(self, config):
        # The output should be a normal node
        config.node_gene_type = DefaultNodeGene
        DefaultGenome.configure_new(self, config)

    def mutate(self, config):
        r = random()
        if r < config.gru_prob:
            config.node_gene_type = GRUNodeGene
        else:
            config.node_gene_type = DefaultNodeGene

        options = list(self.nodes.keys())
        config.__setattr__('read_gate_options', options)
        config.__setattr__('forget_gate_options', options)

        DefaultGenome.mutate(self, config)

        options = list(self.nodes.keys())
        config.__setattr__('read_gate_options', options)
        config.__setattr__('forget_gate_options', options)

        for ng in self.nodes.values():
            if ng is GRUNodeGene:
                ng.mutate_safe()

    def mutate_delete_node(self, config):
        key = DefaultGenome.mutate_delete_node(self, config)
        if key in config.read_gate_options:
            config.read_gate_options.remove(key)
        # Forget gate currently uses the same list
        for node in self.nodes.values():
            if type(node) is GRUNodeGene:
                node.inform_deleted(key, config)

    def configure_crossover(self, genome1, genome2, config):
        DefaultGenome.configure_crossover(self, genome1, genome2, config)
        for key, node in self.nodes.items():
            if type(node) is GRUNodeGene:
                if type(node.read_gate) is int and node.read_gate not in self.nodes.keys():
                    node.read_gate = DEFAULT_READ_GATE
                if type(node.forget_gate) is int and node.forget_gate not in self.nodes.keys():
                    node.forget_gate = DEFAULT_FORGET_GATE


class GRUNetwork(object):
    def __init__(self, inputs, outputs, node_evals, gate_list):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.gate_list = gate_list

        self.values = [{}, {}]
        for v in self.values:
            for k in inputs + outputs:
                v[k] = 0.0

            for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, links, \
                ignored_gates in self.node_evals:
                v[node] = 0.0
                for i, w in links:
                    v[i] = 0.0
        self.active = 0

    def reset(self):
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        t_m1_val = self.values[self.active]
        t_val = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            t_m1_val[i] = v
            t_val[i] = v

        for node, activation, aggregation, bias, response, links, gates in self.node_evals:
            if gates is None:
                node_inputs = [t_m1_val[i] for i, w in links]
                s = aggregation(node_inputs)
                t_val[node] = activation(bias + response * s)
            else:
                rt = gates[0]
                zt = gates[1]

                # Check if gates are assigned to nodes
                if type(rt) is int:
                    rt = t_val[rt]
                if type(zt) is int:
                    zt = t_val[zt]

                node_inputs = [t_m1_val[i] for i, w in links]

                s_c = activation(aggregation(node_inputs))

                t_val[node] = zt * t_m1_val[node] + (1 - zt) * s_c

        return [t_val[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a GRUNetwork). """
        genome_config = config.genome_config
        required = required_for_output(genome_config.input_keys, genome_config.output_keys, genome.connections,
                                       genome.nodes)

        connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]

        node_evals = []

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
        gate_list = set()

        # Add the gates first for proper computation order
        for node_key, inputs in iteritems(node_inputs):
            ng = genome.nodes[node_key]

            if type(ng) is GRUNodeGene:
                for gate_key in [ng.read_gate, ng.forget_gate]:
                    if type(gate_key) is int and gate_key not in gate_list:
                        gate_list = gate_list.union(gate_key)
                        gate_g = genome.nodes[gate_key]
                        activation_function = genome_config.activation_defs.get(gate_g.activation)
                        aggregation_function = genome_config.aggregation_function_defs.get(gate_g.aggregation)
                        node_evals.append(
                            (gate_key, activation_function, aggregation_function, gate_g.bias, gate_g.response, inputs,
                             [gate_g.read_gate, gate_g.forget_gate] if type(gate_g) is GRUNodeGene else None))

        for node_key, inputs in iteritems(node_inputs):
            if node_key not in gate_list:
                ng = genome.nodes[node_key]
                activation_function = genome_config.activation_defs.get(ng.activation)
                aggregation_function = genome_config.aggregation_function_defs.get(ng.aggregation)

                node_evals.append(
                    (node_key, activation_function, aggregation_function, ng.bias, ng.response, inputs,
                     [ng.read_gate, ng.forget_gate] if type(ng) is GRUNodeGene else None))

        nw = GRUNetwork(genome_config.input_keys, genome_config.output_keys, node_evals, gate_list)
        nw.genome = genome
        return nw


class GRUNetworkML(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        self.values = [{}, {}]
        r_idx0 = -len(inputs) - 1
        for v in self.values:
            for k in inputs + outputs:
                v[k] = 0.0

            for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, links, ignored_gates \
                    in self.node_evals:
                v[node] = 0.0
                for i, w in links:
                    if i > 0:  # Not a recurrent edge
                        v[i] = 0.0
                    elif i not in inputs:
                        v[-i+r_idx0] = 0
        self.active = 0

    def reset(self):
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        t_m1_val = self.values[self.active]
        t_val = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            t_m1_val[i] = v
            t_val[i] = v

        r_idx0 = -len(inputs) - 1
        for node, activation, aggregation, bias, response, links, gates in self.node_evals:

            if gates is None:
                node_inputs = [t_m1_val[-i + r_idx0] * w if i <= r_idx0 else t_val[i] for i, w in links]
                s = aggregation(node_inputs)
                t_val[node] = activation(bias + response * s)
            else:
                rt = gates[0]
                zt = gates[1]

                # Check if gates are assigned to nodes
                if type(rt) is int:
                    rt = t_val[rt]
                if type(zt) is int:
                    zt = t_val[zt]

                node_inputs = [t_m1_val[-i + r_idx0] * w * rt if i <= r_idx0 else t_val[i] for i, w in links]

                s_c = activation(aggregation(node_inputs))

                t_val[node] = zt * t_m1_val[node] + (1 - zt) * s_c



        return [t_val[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a GRUNetwork). """
        genome_config = config.genome_config
        connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]

        layers = \
            feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections,
                                genome.nodes)

        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)

                if type(ng) is GRUNodeGene:
                    node_evals.append(
                        (node, activation_function, aggregation_function, ng.bias, ng.response, inputs,
                         [ng.read_gate, ng.forget_gate]))
                else:
                    node_evals.append(
                        (node, activation_function, aggregation_function, ng.bias, ng.response, inputs, None))

        nw = GRUNetworkML(genome_config.input_keys, genome_config.output_keys, node_evals)
        nw.genome = genome
        return nw


def required_for_output(inputs, outputs, connections, nodes):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    :param check_recurrent: check recurrent connections
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.

    Returns a set of identifiers of required nodes.
    """

    required = set(outputs)
    s = set(outputs)
    r_idx0 = -len(inputs) - 1
    while 1:
        # Find nodes not in S whose output is consumed by a node in s.
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:
            break

        gates = set()
        for key in t:
            if key not in inputs:
                node = nodes[-key + r_idx0 if key <= r_idx0 else key]
                if type(node) is GRUNodeGene:
                    if type(node.read_gate) is int:
                        gates.add(node.read_gate)
                    if type(node.forget_gate) is int:
                        gates.add(node.forget_gate)
        t = t.union(gates)

        layer_nodes = set(x for x in t if x not in inputs)

        if not layer_nodes:
            break

        # Replace recurrent id with actual node id
        layer_nodes = set(-x + r_idx0 if x <= r_idx0 else x for x in layer_nodes)

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


def feed_forward_layers(inputs, outputs, connections, nodes):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    :param check_recurrent: check recurrent connections

    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """

    required = required_for_output(inputs, outputs, connections, nodes)

    layers = []
    s = set(inputs)
    while 1:
        # Find candidate nodes c for the next layer.  These nodes should connect
        # a node in s to a node not in s.
        c = set(b for (a, b) in connections if a in s and b not in s)
        # Keep only the used nodes whose entire input set is contained in s.
        t = set()

        r_idx0 = -len(inputs) - 1
        for n in c:
            if n in required and \
                    all((a <= r_idx0) or (a in s) or ((b, a) in connections) for (a, b) in connections if b == n):
                t.add(n)
        if not t:
            break

        layers.append(t)
        s = s.union(t)

    # Check for gates set to dead nodes (not connected to input)
    for key in s:
        if key not in inputs:
            node = nodes[-key + r_idx0 if key <= r_idx0 else key]
            if type(node) is GRUNodeGene:
                if type(node.read_gate) is int and node.read_gate not in s and node.read_gate not in outputs:
                    node.read_gate = DEFAULT_READ_GATE
                if type(node.forget_gate) is int and node.forget_gate not in s and node.forget_gate not in outputs:
                    node.forget_gate = DEFAULT_FORGET_GATE

    return layers