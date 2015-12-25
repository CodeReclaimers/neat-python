class IFSynapse(object):
    """ A synapse indicates the connection strength between two neurons (or itself) """

    def __init__(self, source, dest, weight):
        self.__weight = weight
        self.__source = source
        self.__dest = dest

    def advance(self):
        """Advances time in 1 ms."""
        if self.__source.has_fired:
            self.__dest.current += self.__weight * self.__source.output


class IFNetwork(object):
    """ A neural network has a list of neurons linked by synapses """

    def __init__(self, neurons, input_neurons, output_neurons, synapses):
        self.neurons = neurons
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.synapses = synapses

    def __repr__(self):
        return '{0:d} nodes and {1:d} synapses'.format(len(self.neurons), len(self.synapses))

    def advance(self, inputs):
        assert len(inputs) == len(self.input_neurons), "Wrong number of inputs."
        for i, n in zip(inputs, self.input_neurons):
            n.current = n.bias + i
        for s in self.synapses:
            s.advance()
        for n in self.neurons.values():
            n.advance()
        return [n.has_fired for n in self.output_neurons]

    def reset(self):
        """Resets the network's state."""
        for n in self.neurons.values():
            n.reset()


class IFNeuron(object):
    """Neuron based on the integrate and fire model"""

    def __init__(self, bias=0, tau=10, vrest=-70, vreset=-70, vt=-55):
        """
        tau, vrest, vreset, vthreshold are the parameters of this model.
        tau: membrane time constant in ms.
        vrest: rest potential in mV.
        vreset: reset potential in mV.
        vt: firing threshold in mV.
        """
        self.__invtau = 1.0 / tau
        self.__vrest = vrest
        self.__vreset = vreset
        self.__vt = vt
        self.__bias = bias
        self.__v = self.__vreset
        assert self.__v < self.__vt
        self.__has_fired = False
        self.current = self.__bias

    def advance(self):
        """Advances time in 1 ms."""
        self.__v += self.__invtau * (self.__vrest - self.__v + self.current)
        if self.__v >= self.__vt:
            self.__has_fired = True
            self.__v = self.__vreset
        else:
            self.__has_fired = False
        self.current = self.__bias

    def reset(self):
        """Resets all state variables."""
        self.__v = self.__vreset
        self.__has_fired = False
        self.current = self.__bias

    potential = property(lambda self: self.__v, doc='Membrane potential')
    has_fired = property(lambda self: self.__has_fired,
                         doc='Indicates whether the neuron has fired')


def create_phenotype(genome):
    """ Receives a genome and returns its phenotype (a neural network) """

    neurons = {}
    input_neurons = []
    output_neurons = []
    for ng in genome.node_genes.values():
        neurons[ng.ID] = IFNeuron(ng.bias)
        if ng.type == 'INPUT':
            input_neurons.append(neurons[ng.ID])
        elif ng.type == 'OUTPUT':
            output_neurons.append(neurons[ng.ID])

    synapses = [IFSynapse(neurons[cg.in_node_id], neurons[cg.out_node_id], cg.weight)
                for cg in genome.conn_genes if cg.enabled]

    return IFNetwork(neurons, input_neurons, output_neurons, synapses)
