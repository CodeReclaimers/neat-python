"""
 Module for Continuous-Time Recurrent Neural Networks.
 This pure python version solves the differential equations
 using a simple Forward-Euler method. For a higher precision
 method, use the C++ extension with 4th order Runge-Kutta.
"""
from neat.nn import nn_pure as nn

class CTNeuron(nn.Neuron):
    """ Continuous-time neuron model based on:

        Beer, R. D. and Gallagher, J.C. (1992).
        Evolving Dynamical Neural Networks for Adaptive Behavior.
        Adaptive Behavior 1(1):91-122.
    """
    def __init__(self, neurontype, id = None, bias = 0.0, response = 1.0, activation_type = 'exp', tau = 1.0):
        super(CTNeuron, self).__init__(neurontype, id, bias, response, activation_type)

        # decay rate
        self.__tau  = tau
        # needs to set the initial state (initial condition for the ODE)
        self.__state = 0.1 #TODO: Verify what's the "best" initial state
        # fist output
        self._output = nn.sigmoid(self.__state + self._bias, self._response, self._activation_type)
        # integration step
        self.__dt = 0.05 # depending on the tau constant, the integration step must
                         # be adjusted accordingly to avoid numerical instability

    def set_integration_step(self, step):
        self.__dt = step

    def set_init_state(self, state):
        self.__state = state
        self._output = nn.sigmoid(self.__state + self._bias, self._response, self._activation_type)

    def activate(self):
        """ Updates neuron's state for a single time-step. . """
        assert self._type is not 'INPUT'
        self.__update_state()
        return nn.sigmoid(self.__state + self._bias, self._response, self._activation_type)

    def __update_state(self):
        """ Returns neuron's next state using Forward-Euler method. """
        self.__state += self.__dt*(1.0/self.__tau)*(-self.__state + self._update_activation())


def create_phenotype(chromo):
    """ Receives a chromosome and returns its phenotype (a CTRNN). """
    neurons_list = [CTNeuron(ng._type,
                             ng._id,
                             ng._bias,
                             ng._response,
                             ng._activation_type,
                             ng._time_constant) \
                    for ng in chromo._node_genes]

    conn_list = [(cg.innodeid, cg.outnodeid, cg.weight) \
                  for cg in chromo.conn_genes if cg.enabled]

    return nn.Network(neurons_list, conn_list, chromo.sensors)

if __name__ == "__main__":
    # This example follows from Beer's C++ source code available at:
    # http://mypage.iu.edu/~rdbeer/

    # create two output neurons (they won't receive any external inputs)
    N1 = CTNeuron('OUTPUT', 1, -2.75, 1.0, 'exp', 0.5)
    N2 = CTNeuron('OUTPUT', 2, -1.75, 1.0, 'exp', 0.5)
    N1.set_init_state(-0.084000643)
    N2.set_init_state(-0.408035109)

    neurons_list = [N1, N2]
    # create some synapses
    conn_list = [(1, 1, 4.5), (1, 2, -1.0), (2, 1, 1.0), (2, 2, 4.5)]
    # create the network
    net = nn.Network(neurons_list, conn_list)
    # activates the network
    print "%.17f %.17f" %(N1._output, N2._output)
    for i in xrange(1000):
        #print net.pactivate()
        output = net.pactivate()
        print "%.17f %.17f" %(output[0], output[1])


