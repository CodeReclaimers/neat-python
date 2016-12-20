from neat.graphs import feed_forward_layers
from neat.six_util import itervalues


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

    @staticmethod
    def create(genome, config):
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


