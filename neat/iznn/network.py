from iznn_pure import Neuron, Synapse


class Network(object):
    """ A neural network has a list of neurons linked by synapses """

    def __init__(self, neurons=None, input_neurons=None, output_neurons=None, synapses=None):
        if neurons is None:
            self.__neurons = {}
        else:
            self.__neurons = neurons

        if input_neurons is None:
            self.__input_neurons = []
        else:
            self.__input_neurons = input_neurons

        if output_neurons is None:
            self.__output_neurons = []
        else:
            self.__output_neurons = output_neurons

        if synapses is None:
            self.__synapses = []
        else:
            self.__synapses = synapses

    def __repr__(self):
        return '%d nodes and %d synapses' % (len(self.__neurons), len(self.__synapses))

    def advance(self, inputs):
        assert len(inputs) == len(self.__input_neurons), "Wrong number of inputs."
        for i, input_index in enumerate(inputs):
            self.__input_neurons[i].current += input_index
        for s in self.__synapses:
            s.advance()
        for n in self.__neurons.values():
            n.advance()
        return [n.has_fired for n in self.__output_neurons]

    def reset(self):
        """Resets the network's state."""
        for n in self.__neurons.values():
            n.reset()

    neurons = property(lambda self: self.__neurons.values())


def create_phenotype(chromosome):
    """ Receives a chromosome and returns its phenotype (a neural network) """

    neurons = {}
    input_neurons = []
    output_neurons = []
    for ng in chromosome.node_genes:
        neurons[ng.ID] = Neuron(ng.bias)
        if ng.type == 'INPUT':
            input_neurons.append(neurons[ng.ID])
        elif ng.type == 'OUTPUT':
            output_neurons.append(neurons[ng.ID])

    synapses = [Synapse(neurons[cg.innodeid], neurons[cg.outnodeid], cg.weight)
                for cg in chromosome.conn_genes.values() if cg.enabled]

    return Network(neurons, input_neurons, output_neurons, synapses)
