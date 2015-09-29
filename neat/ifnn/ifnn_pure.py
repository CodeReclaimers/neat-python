from neat.iznn.iznn_pure import Synapse
from neat.iznn.network import Network

class Neuron(object):
    """Neuron based on the integrate and fire model"""
    def __init__(self, bias = 0, tau = 10, vrest = -70, vreset = -70, vt = -55):
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
    
    potential = property(lambda self: self.__v, doc = 'Membrane potential')
    has_fired = property(lambda self: self.__has_fired,
                     doc = 'Indicates whether the neuron has fired')

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
