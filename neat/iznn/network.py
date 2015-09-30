from iznn_pure import *

class Network(object):
    """ A neural network has a list of neurons linked by synapses """
    def __init__(self, neurons=[], input_neurons = [], output_neurons = [], synapses=[]):
        self.__neurons = neurons
        self.__input_neurons = input_neurons
        self.__output_neurons = output_neurons
        self.__synapses = synapses

    def __repr__(self):
        return '%d nodes and %d synapses' % (len(self.__neurons), len(self.__synapses))

    def advance(self, inputs):
        assert len(inputs) == len(self.__input_neurons), "Wrong number of inputs."
        for i, input in enumerate(inputs):
            self.__input_neurons[i].current += input
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
        neurons[ng.id] = Neuron(ng.bias)
        if ng.type == 'INPUT':
            input_neurons.append(neurons[ng.id])
        elif ng.type == 'OUTPUT':
            output_neurons.append(neurons[ng.id])

    synapses = [Synapse(neurons[cg.innodeid], neurons[cg.outnodeid], cg.weight) \
                 for cg in chromosome.conn_genes if cg.enabled]

    return Network(neurons, input_neurons, output_neurons, synapses)

if __name__ == '__main__':
    from neat import visualize
    n = Neuron(10)
    spike_train = []
    for i in range(1000):
        spike_train.append(n.potential)
        print '%d\t%f' % (i, n.potential)
        n.advance()
        
    visualize.plot_spikes(spike_train)
