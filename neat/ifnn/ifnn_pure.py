import math

class Neuron(object):
    'Neuron based on the integrate and fire model'
    def __init__(self, bias = 0, tau = 10, vrest = -70, vreset = -70, vt = -55):
        '''
        tau, vrest, vreset, vthreshold are the parameters of this model.
        tau: membrane time constant in ms.
        vrest: rest potential in mV.
        vreset: reset potential in mV.
        vt: firing threshold in mV.
        '''
        self.__invtau = 1.0 / tau
        self.__vrest = vrest
        self.__vreset = vreset
        self.__vt = vt
        self.__bias = 0
        self.__v = self.__vreset
        assert self.__v < self.__vt
        self.__has_fired = False
    
    def advance(self):
        'Advances time in 1 ms.'
        self.__v = self.__vrest + (1 -  self.__invtau) * (self.__v - self.__vrest) + \
            self.__invtau * self.current
        if self.__v >= self.__vt:
            self.__has_fired = True
            self.__v = self.__vreset
        else:
            self.__has_fired = False
        self.current = self.__bias
    
    def reset(self):
        'Resets all state variables.'
        self.__v = self.__vreset
        self.__has_fired = False
        self.current = self.__bias
    
    potential = property(lambda self: self.__v, doc = 'Membrane potential')
    has_fired = property(lambda self: self.__has_fired,
                     doc = 'Indicates whether the neuron has fired')
