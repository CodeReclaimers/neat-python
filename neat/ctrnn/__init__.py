"""
 Module for Continuous-Time Recurrent Neural Networks.
 This pure python version solves the differential equations
 using the forward Euler method.
"""
import random
from neat.indexer import Indexer
from neat import activation_functions
from neat.genes import NodeGene


class CTNodeGene(NodeGene):
    """ Continuous-time node gene - used in CTRNNs.
        The main difference here is the addition of
        a decay rate given by the time constant.
    """
    def __init__(self, ID, node_type, bias=0.0, response=4.924273,
                 activation_type='sigmoid', time_constant=1.0):
        super(CTNodeGene, self).__init__(ID, node_type, bias, response, activation_type)
        self.time_constant = time_constant

    def mutate(self, config):
        super(CTNodeGene, self).mutate(config)
        # TODO: There is no support for prob_mutate_time_constant in the Config
        # class, so currently the user must add it themselves.
        if hasattr(config, 'prob_mutate_time_constant') and \
            random.random() < config.prob_mutate_time_constant:
            self.mutate_time_constant(config)

    def mutate_time_constant(self, config):
        """ Warning: perturbing the time constant (tau) may result in numerical instability """
        self.time_constant *= random.gauss(1.0, 0.01)
        if self.time_constant < 0:
            self.time_constant = 0
        return self

    def get_child(self, other):
        """ Creates a new NodeGene randomly inheriting its attributes from parents """
        assert (self.ID == other.ID)

        ng = CTNodeGene(self.ID, self.type,
                        random.choice((self.bias, other.bias)),
                        random.choice((self.response, other.response)),
                        random.choice((self.activation_type, other.activation_type)),
                        random.choice((self.time_constant, other.time_constant)))
        return ng

    def __str__(self):
        return 'CTNodeGene(id={0}, type={1}, bias={2}, response={3}, activation={4}, time_constant={5})'.format(
            self.ID, self.type, self.bias, self.response, self.activation_type, self.time_constant)

    def copy(self):
        return CTNodeGene(self.ID, self.type, self.bias,
                          self.response, self.activation_type, self.time_constant)


class Neuron(object):
    """ A simple sigmoidal neuron """
    # TODO: Get rid of these global indexers.
    indexer = Indexer(0)

    def __init__(self, neuron_type, ID, bias, response, activation_type):
        assert neuron_type in ('INPUT', 'OUTPUT', 'HIDDEN')

        self.type = neuron_type
        self.ID = self.indexer.get_next(ID)
        self.bias = bias
        self.response = response
        self.activation = activation_functions.get(activation_type)

        self._synapses = []
        self.output = 0.0  # for recurrent networks all neurons must have an "initial state"

    def activate(self):
        """Activates the neuron"""
        assert self.type is not 'INPUT'
        z = self.bias + self.response * self._update_activation()
        return self.activation(z)

    def _update_activation(self):
        soma = 0.0
        for s in self._synapses:
            soma += s.incoming()
        return soma

    def create_synapse(self, s):
        self._synapses.append(s)

    def __repr__(self):
        return '{0:d} {1!s}'.format(self.ID, self.type)


class CTNeuron(Neuron):
    """ Continuous-time neuron model based on:

        Beer, R. D. and Gallagher, J.C. (1992).
        Evolving Dynamical Neural Networks for Adaptive Behavior.
        Adaptive Behavior 1(1):91-122.
    """

    def __init__(self, neuron_type, ID, bias, response, activation_type, tau):
        super(CTNeuron, self).__init__(neuron_type, ID, bias, response, activation_type)

        # decay rate
        self.__tau = tau
        # needs to set the initial state (initial condition for the ODE)
        self.__state = 0.1
        # first output
        z = self.bias + self.response * self.__state
        self._output = self.activation(z)
        # integration step
        self.__dt = 0.05  # depending on the tau constant, the integration step must
        # be adjusted accordingly to avoid numerical instability

    def set_integration_step(self, step):
        self.__dt = step

    def set_init_state(self, state):
        self.__state = state

        self._output = self.activation(self.bias + self.response * self.__state)

    def activate(self):
        """ Updates neuron's state for a single time-step. . """
        assert self.type is not 'INPUT'
        self.__update_state()
        return self.activation(self.bias + self.response * self.__state)

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
        return '{0!s} -> {1!s} -> {2!s}'.format(self.source.ID, self.weight, self.destination.ID)


class Network(object):
    """A neural network has a list of neurons linked by synapses"""

    def __init__(self, neurons, links, num_inputs):
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

    def set_integration_step(self, step):
        for neuron in self.neurons:
            neuron.set_integration_step(step)

    def reset(self):
        for neuron in self.neurons:
            neuron.output = 0.0

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def add_synapse(self, synapse):
        self.synapses.append(synapse)

    def __repr__(self):
        return '{0:d} nodes and {1:d} synapses'.format(len(self.neurons), len(self.synapses))

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
            if n.type != 'INPUT':  # hidden or output neurons
                current_state.append(n.activate())
        # updates all neurons at once
        net_output = []
        for n, state in zip(self.neurons[self._num_inputs:], current_state):
            n.output = state  # updates from the previous step
            if n.type == 'OUTPUT':
                net_output.append(n.output)

        return net_output


def create_phenotype(genome):
    """ Receives a genome and returns its phenotype (a continuous-time recurrent neural network). """
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
