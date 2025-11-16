from neat.graphs import feed_forward_layers
import random

class FeedForwardNetwork:
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = {key: 0.0 for key in inputs + outputs}

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError(f"Expected {len(self.input_nodes):n} inputs, got {len(inputs):n}")

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)

        return [self.values[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config, unique_value=False, random_values=False):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers, required = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        # Input nodes are not in 'required', but we need to check connections from them too
        required_with_inputs = required.union(set(config.genome_config.input_keys))
        for layer in layers:
            for node in layer:
                inputs = []
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node and inode in required_with_inputs:
                        cg = genome.connections[conn_key]
                        if random_values:
                            cg.weight = random.uniform(-1.0, 1.0)
                        if unique_value:
                            cg.weight = unique_value
                        inputs.append((inode, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)
