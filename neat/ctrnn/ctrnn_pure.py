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

    def __init__(self, neurontype, id=None, bias=0.0, response=1.0, activation_type='exp', tau=1.0):
        super(CTNeuron, self).__init__(neurontype, id, bias, response, activation_type)

        # decay rate
        self.__tau = tau
        # needs to set the initial state (initial condition for the ODE)
        self.__state = 0.1  # TODO: Verify what's the "best" initial state
        # fist output
        self._output = nn.sigmoid(self.__state + self._bias, self._response, self._activation_type)
        # integration step
        self.__dt = 0.05  # depending on the tau constant, the integration step must
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
        self.__state += self.__dt * (1.0 / self.__tau) * (-self.__state + self._update_activation())


def create_phenotype(chromo):
    """ Receives a chromosome and returns its phenotype (a CTRNN). """
    neurons_list = [CTNeuron(ng.type,
                             ng.ID,
                             ng.bias,
                             ng.response,
                             ng.activation_type,
                             ng.time_constant) \
                    for ng in chromo.node_genes]

    conn_list = [(cg.innodeid, cg.outnodeid, cg.weight) \
                 for cg in chromo.conn_genes.values() if cg.enabled]

    return nn.Network(neurons_list, conn_list, chromo.sensors)
