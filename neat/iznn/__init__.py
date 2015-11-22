# -*- coding: UTF-8 -*-
"""
This module implements a spiking neural network.
Neurons are based on the model described by:
    
Izhikevich, E. M.
Simple Model of Spiking Neurons
IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 14, NO. 6, NOVEMBER 2003
"""


class Neuron(object):
    def __init__(self, bias=0, a=0.02, b=0.2, c=-65.0, d=8.0):
        """
        a, b, c, d are the parameters of this model.
        a: the time scale of the recovery variable.
        b: the sensitivity of the recovery variable.
        c: the after-spike reset value of the membrane potential.
        d: after-spike reset of the recovery variable.

        The following parameters produce some known spiking behaviors:

        Regular spiking: a = 0.02, b = 0.2, c = -65.0, d = 8.0
        Intrinsically bursting: a = 0.02, b = 0.2, c = -55.0, d = 4.0
        Chattering: a = 0.02, b = 0.2, c = -50.0, d = 2.0
        Fast spiking: a = 0.1, b = 0.2, c = -65.0, d = 2.0
        Thalamo-cortical: a = 0.02, b = 0.25, c = -65.0, d = 0.05
        Resonator: a = 0.1, b = 0.25, c = -65.0, d = 2.0
        Low-threshold spiking: a = 0.02, b = 0.25, c = -65, d = 2.0
        """
        self.__a = a
        self.__b = b
        self.__c = c
        self.__d = d
        self.__v = self.__c  # membrane potential
        self.__u = self.__b * self.__v  # membrane recovery variable
        self.__has_fired = False
        self.__bias = bias
        self.current = self.__bias

    def advance(self):
        """Advances time in 1 ms.

        v' = 0.04 * v^2 + 5v + 140 - u + I
        u' = a * (b * v - u)

        if v >= 30 then v <- c, u <- u + d
        """
        self.__v += 0.5 * (0.04 * self.__v ** 2 + 5 * self.__v + 140 - self.__u + self.current)
        self.__v += 0.5 * (0.04 * self.__v ** 2 + 5 * self.__v + 140 - self.__u + self.current)
        self.__u += self.__a * (self.__b * self.__v - self.__u)
        if self.__v > 30:
            self.__has_fired = True
            self.__v = self.__c
            self.__u += self.__d
        else:
            self.__has_fired = False
        self.current = self.__bias

    def reset(self):
        """Resets all state variables."""
        self.__v = self.__c
        self.__u = self.__b * self.__v
        self.__has_fired = False
        self.current = self.__bias

    potential = property(lambda self: self.__v, doc='Membrane potential')
    recovery = property(lambda self: self.__u, doc='Membrane recovery')
    has_fired = property(lambda self: self.__has_fired,
                         doc='Indicates whether the neuron has fired')


class Synapse(object):
    """ A synapse indicates the connection strength between two neurons (or itself) """

    def __init__(self, source, dest, weight):
        self.__weight = weight
        self.__source = source
        self.__dest = dest

    def advance(self):
        """Advances time in 1 ms."""
        if self.__source.has_fired:
            self.__dest.current += self.__weight


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
        for i, n in zip(inputs, self.__input_neurons):
            n.current += i
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
    for ng in chromosome.node_genes.values():
        # TODO: It seems like we should have a separate node gene implementation
        # that encodes more (all?) of the Izhikevitch model parameters.
        neurons[ng.ID] = Neuron(ng.bias)
        if ng.type == 'INPUT':
            input_neurons.append(neurons[ng.ID])
        elif ng.type == 'OUTPUT':
            output_neurons.append(neurons[ng.ID])

    synapses = [Synapse(neurons[cg.in_node_id], neurons[cg.out_node_id], cg.weight)
                for cg in chromosome.conn_genes.values() if cg.enabled]

    return Network(neurons, input_neurons, output_neurons, synapses)
