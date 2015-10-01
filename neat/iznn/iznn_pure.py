# -*- coding: UTF-8 -*-

class Neuron(object):
    """
    A spiking neuron model based on:

    Izhikevich, E. M.
    Simple Model of Spiking Neurons
    IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 14, NO. 6, NOVEMBER 2003
    """

    def __init__(self, bias = 0, a = 0.02, b = 0.2, c = -65.0, d = 8.0):
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
        self.__v = self.__c # membrane potential
        self.__u = self.__b * self.__v # membrane recovery variable
        self.__has_fired = False
        self.__bias = bias
        self.current = self.__bias

    def advance(self):
        """Advances time in 1 ms."""
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

    potential = property(lambda self: self.__v, doc = 'Membrane potential')
    has_fired = property(lambda self: self.__has_fired,
                     doc = 'Indicates whether the neuron has fired')

class Synapse:
    """ A synapse indicates the connection strength between two neurons (or itself) """
    def __init__(self, source, dest, weight):
        self.__weight = weight
        self.__source = source
        self.__dest = dest

    def advance(self):
        """Advances time in 1 ms."""
        if self.__source.has_fired:
            self.__dest.current += self.__weight # dest.current or dest.__v ?


if __name__ == '__main__':
    from neat import visualize
    n = Neuron(10)
    neuron_membrane = []
    for i in range(1000):
        neuron_membrane.append(n.potential)
        #print '%d\t%f' % (i, n.potential)
        n.advance()

    visualize.plot_spikes(neuron_membrane, view=True)
