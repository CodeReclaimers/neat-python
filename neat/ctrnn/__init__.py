"""
 Module for Continuous-Time Recurrent Neural Networks.
 This pure python version solves the differential equations
 using a simple Forward-Euler method. For a higher precision
 method, use the C++ extension with 4th order Runge-Kutta.
"""
from neat.indexer import Indexer
from neat.nn import exp_sigmoid, tanh_sigmoid


class Neuron(object):
    """ A simple sigmoidal neuron """
    _indexer = Indexer(0)

    def __init__(self, neuron_type, ID, bias, response, activation_type):
        assert neuron_type in ('INPUT', 'OUTPUT', 'HIDDEN')
        assert activation_type in ('exp', 'tanh')

        self.type = neuron_type
        self.ID = self._indexer.next(ID)  # every neuron has an ID
        self.bias = bias
        self.response = response
        if activation_type == 'exp':
            self.activation = exp_sigmoid
        elif activation_type == 'tanh':
            self.activation = tanh_sigmoid

        self._synapses = []
        self.output = 0.0  # for recurrent networks all neurons must have an "initial state"

    def activate(self):
        """Activates the neuron"""
        assert self.type is not 'INPUT'
        return self.activation(self.bias, self.response, self._update_activation())

    def _update_activation(self):
        soma = 0.0
        for s in self._synapses:
            soma += s.incoming()
        return soma

    def create_synapse(self, s):
        self._synapses.append(s)

    def __repr__(self):
        return '%d %s' % (self.ID, self.type)


class CTNeuron(Neuron):
    """ Continuous-time neuron model based on:

        Beer, R. D. and Gallagher, J.C. (1992).
        Evolving Dynamical Neural Networks for Adaptive Behavior.
        Adaptive Behavior 1(1):91-122.
    """

    def __init__(self, neuron_type, ID=None, bias=0.0, response=1.0, activation_type='exp', tau=1.0):
        super(CTNeuron, self).__init__(neuron_type, ID, bias, response, activation_type)

        # decay rate
        self.__tau = tau
        # needs to set the initial state (initial condition for the ODE)
        self.__state = 0.1  # TODO: Verify what's the "best" initial state
        # fist output
        self._output = self.activation(self.bias, self.response, self.__state)
        # integration step
        self.__dt = 0.05  # depending on the tau constant, the integration step must
        # be adjusted accordingly to avoid numerical instability

    def set_integration_step(self, step):
        self.__dt = step

    def set_init_state(self, state):
        self.__state = state
        self._output = self.activation(self.bias, self.response, self.__state)

    def activate(self):
        """ Updates neuron's state for a single time-step. . """
        assert self.type is not 'INPUT'
        self.__update_state()
        return self.activation(self.bias, self.response, self.__state)

    def __update_state(self):
        """ Returns neuron's next state using Forward-Euler method. """
        self.__state += self.__dt * (1.0 / self.__tau) * (-self.__state + self._update_activation())

class Synapse(object):
    """A synapse indicates the connection strength between two neurons (or itself)"""

    def __init__(self, source, destination, weight):
        self.weight = weight
        self.source = source  # a 'pointer' to the source neuron
        self.destination = destination  # a 'pointer' to the destination neuron
        destination.create_synapse(self)  # adds the synapse to the destination neuron

    def incoming(self):
        """ Receives the incoming signal from a sensor or another neuron
            and returns the value to the neuron it belongs to. """
        return self.weight * self.source.output

    def __repr__(self):
        return '%s -> %s -> %s' % (self.source.ID, self.weight, self.destination.ID)


class Network(object):
    """A neural network has a list of neurons linked by synapses"""

    def __init__(self, neurons=None, links=None, num_inputs=0):
        if not neurons:
            neurons = []
        self.neurons = neurons
        self.synapses = []
        self._num_inputs = num_inputs

        if links is not None:
            nodes = {}  # a temporary dictionary to create the network connections
            for n in self.neurons:
                nodes[n.ID] = n
            for c in links:
                self.synapses.append(Synapse(nodes[c[0]], nodes[c[1]], c[2]))

    def flush(self):
        for neuron in self.neurons:
            neuron.output = 0.0

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def add_synapse(self, synapse):
        self.synapses.append(synapse)

    def __repr__(self):
        return '%d nodes and %d synapses' % (len(self.neurons), len(self.synapses))

    def serial_activate(self, inputs):
        """Serial (asynchronous) network activation method. Mostly
           used  in classification tasks (supervised learning) in
           feedforward topologies. All neurons are updated (activated)
           one at a time following their order of importance, so if
           you're defining your own feedforward topology, make sure
           you got them in the right order of activation.
        """
        assert len(inputs) == self._num_inputs, "Wrong number of inputs."

        # assign "input neurons'" output values (sensor readings)
        input_neurons = [n for n in self.neurons[:self._num_inputs] if n.type == 'INPUT']
        for v, n in zip(inputs, input_neurons):
            n.output = v

        # activate all neurons in the network (except for the inputs)
        net_output = []
        for n in self.neurons[self._num_inputs:]:
            n.output = n.activate()
            if n.type == 'OUTPUT':
                net_output.append(n.output)
        return net_output

    def parallel_activate(self, inputs=None):
        """Parallel (synchronous) network activation method. Mostly used
           for control and unsupervised learning (i.e., artificial life)
           in recurrent networks. All neurons are updated (activated)
           simultaneously.
        """
        if inputs is not None:
            assert len(inputs) == self._num_inputs, "Wrong number of inputs."

            # assign "input neurons'" output values (sensor readings)
            input_neurons = [n for n in self.neurons[:self._num_inputs] if n.type == 'INPUT']
            for v, n in zip(inputs, input_neurons):
                n.output = v

        # the current state is like a "photograph" taken at each time step
        # representing all neuron's state at that time (think of it as a clock)
        current_state = []
        for n in self.neurons:
            if n.type != 'INPUT': # hidden or output neurons
                current_state.append(n.activate())
        # updates all neurons at once
        net_output = []
        for n, state in zip(self.neurons[self._num_inputs:], current_state):
            n.output = state  # updates from the previous step
            if n.type == 'OUTPUT':
                net_output.append(n.output)

        return net_output


def create_phenotype(genome):
    """ Receives a genome and returns its phenotype (a CTRNN). """
    neurons_list = [CTNeuron(ng.type,
                             ng.ID,
                             ng.bias,
                             ng.response,
                             ng.activation_type,
                             ng.time_constant)
                    for ng in genome.node_genes.values()]

    conn_list = [(cg.in_node_id, cg.out_node_id, cg.weight)
                 for cg in genome.conn_genes.values() if cg.enabled]

    return Network(neurons_list, conn_list, genome.num_inputs)
