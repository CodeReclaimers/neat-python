import math
import random


def sigmoid(x, response, activation_type):
    """ Sigmoidal type of activation function """
    output = 0
    try:
        if activation_type == 'exp':
            if x < - 30:
                output = 0.0
            elif x > 30:
                output = 1.0
            else:
                output = 1.0 / (1.0 + math.exp(-x * response))
        elif activation_type == 'tanh':
            if x < - 20:
                output = -1.0
            elif x > 20:
                output = +1.0
            else:
                output = math.tanh(x * response)
        else:
            # raise exception
            print 'Invalid activation type selected:', activation_type
            # raise NameError('Invalid activation type selected:', activation_type)

    except OverflowError:
        print 'Overflow error: x = ', x

    return output


class Neuron(object):
    """ A simple sigmoidal neuron """
    __next_id = 1

    @classmethod
    def __get_next_id(cls, ID):
        if ID is None:
            ID = cls.__next_id
            cls.__next_id += 1

        return ID

    def __init__(self, neurontype, ID=None, bias=0.0, response=1.0, activation_type='exp'):

        self.ID = Neuron.__get_next_id(ID)  # every neuron has an ID

        self._synapses = []

        self.bias = bias
        self.type = neurontype
        assert (self.type in ('INPUT', 'OUTPUT', 'HIDDEN'))

        self.activation_type = activation_type  # default is exponential

        self.response = response  # default = 4.924273 (Stanley, p. 146)
        self.output = 0.0  # for recurrent networks all neurons must have an "initial state"

    def activate(self):
        """Activates the neuron"""
        assert self.type is not 'INPUT'
        return sigmoid(self._update_activation() + self.bias, self.response, self.activation_type)

    def _update_activation(self):
        soma = 0.0
        for s in self._synapses:
            soma += s.incoming()
        return soma

    def create_synapse(self, s):
        self._synapses.append(s)

    def __repr__(self):
        return '%d %s' % (self.ID, self.type)


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

    def sactivate(self, inputs=None):
        """Serial (asynchronous) network activation method. Mostly
           used  in classification tasks (supervised learning) in
           feedforward topologies. All neurons are updated (activated)
           one at a time following their order of importance, so if
           you're defining your own feedforward topology, make sure
           you got them in the right order of activation.
        """
        if not inputs:
            inputs = []
        assert len(inputs) == self._num_inputs, "Wrong number of inputs."
        # assign "input neurons'" output values (sensor readings)

        it = iter(inputs)
        for n in self.neurons[:self._num_inputs]:
            if n.type == 'INPUT':
                n.output = it.next()  # iterates over inputs
        # activate all neurons in the network (except for the inputs)
        net_output = []
        for n in self.neurons[self._num_inputs:]:
            n.output = n.activate()
            if n.type == 'OUTPUT':
                net_output.append(n.output)
        return net_output

    def pactivate(self, inputs=None):
        """Parallel (synchronous) network activation method. Mostly used
           for control and unsupervised learning (i.e., artificial life)
           in recurrent networks. All neurons are updated (activated)
           simultaneously.
        """
        if not inputs:
            inputs = []
        assert len(inputs) == self._num_inputs, "Wrong number of inputs."

        # the current state is like a "photograph" taken at each time step
        # representing all neuron's state at that time (think of it as a clock)
        current_state = []
        it = iter(inputs)
        for n in self.neurons:
            if n.type == 'INPUT':
                n.output = it.next()  # iterates over inputs
            else:  # hidden or output neurons
                current_state.append(n.activate())
        # updates all neurons at once
        net_output = []
        for n, state in zip(self.neurons[self._num_inputs:], current_state):
            n.output = state  # updates from the previous step
            if n.type == 'OUTPUT':
                net_output.append(n.output)

        return net_output


class FeedForward(Network):
    """ A feedforward network is a particular class of neural network.
        Only one hidden layer is considered for now.
    """

    def __init__(self, layers, use_bias=False, activation_type=None):
        super(FeedForward, self).__init__()

        self.__input_layer = layers[0]
        self.__output_layer = layers[-1]
        self.__hidden_layers = layers[1:-1]
        self.__use_bias = use_bias

        self._num_inputs = layers[0]
        self.__create_net(activation_type)

    def __create_net(self, activation_type):

        # assign random weights for bias
        if self.__use_bias:
            r = random.uniform
        else:
            r = lambda a, b: 0

        for i in xrange(self.__input_layer):
            self.add_neuron(Neuron('INPUT'))

        for i in xrange(self.__hidden_layers[0]):
            self.add_neuron(Neuron('HIDDEN', bias=r(-1, 1),
                                   response=1,
                                   activation_type=activation_type))

        for i in xrange(self.__output_layer):
            self.add_neuron(Neuron('OUTPUT', bias=r(-1, 1),
                                   response=1,
                                   activation_type=activation_type))

        r = random.uniform  # assign random weights
        # inputs -> hidden
        for i in self.neurons[:self.__input_layer]:
            for h in self.neurons[self.__input_layer:-self.__output_layer]:
                self.add_synapse(Synapse(i, h, r(-1, 1)))
        # hidden -> outputs
        for h in self.neurons[self.__input_layer:-self.__output_layer]:
            for o in self.neurons[-self.__output_layer:]:
                self.add_synapse(Synapse(h, o, r(-1, 1)))


def create_phenotype(chromo):
    """ Receives a chromosome and returns its phenotype (a neural network) """

    neurons_list = [Neuron(ng.type, ng.ID,
                           ng.bias,
                           ng.response,
                           ng.activation_type)
                    for ng in chromo.node_genes.values()]

    conn_list = [(cg.in_node_id, cg.out_node_id, cg.weight)
                 for cg in chromo.conn_genes.values() if cg.enabled]

    return Network(neurons_list, conn_list, chromo.num_inputs)


def create_ffphenotype(chromo):
    """ Receives a chromosome and returns its phenotype (a neural network) """

    # first create inputs
    neurons_list = [Neuron('INPUT', ng.ID, 0, 0) \
                    for ng in chromo.node_genes.values() if ng.type == 'INPUT']

    # Add hidden nodes in the right order
    for ID in chromo.node_order:
        gene = chromo.node_genes[ID]
        neurons_list.append(Neuron('HIDDEN', ID, gene.bias, gene.response, gene.activation_type))
    # finally the output
    neurons_list.extend(Neuron('OUTPUT', ng.ID, ng.bias,
                               ng.response, ng.activation_type) \
                        for ng in chromo.node_genes.values() if ng.type == 'OUTPUT')

    assert (len(neurons_list) == len(chromo.node_genes))

    conn_list = [(cg.in_node_id, cg.out_node_id, cg.weight) \
                 for cg in chromo.conn_genes.values() if cg.enabled]

    return Network(neurons_list, conn_list, chromo.num_inputs)
