"""
This module implements a spiking neural network.
Neurons are based on the model described by:
    
Izhikevich, E. M.
Simple Model of Spiking Neurons
IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 14, NO. 6, NOVEMBER 2003
"""


class Neuron(object):
    def __init__(self, bias, a, b, c, d, time_step_msec=1.0):
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
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.bias = bias
        self.dt_msec = time_step_msec

        # Membrane potential (millivolts).
        self.v = self.c

        # Membrane recovery variable.
        self.u = self.b * self.v

        self.output = 0.0
        self.current = self.bias

    def advance(self):
        """
        Advances simulation time by 1 ms.

        v' = 0.04 * v^2 + 5v + 140 - u + I
        u' = a * (b * v - u)

        if v >= 30 then
            v <- c, u <- u + d
        """
        # TODO: Make the time step adjustable, and choose an appropriate
        # numerical integration method to maintain stability.
        # TODO: The need to catch overflows indicates that the current method is
        # not stable for all possible network configurations and states.
        try:
            self.v += 0.5 * self.dt_msec * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + self.current)
            self.v += 0.5 * self.dt_msec * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + self.current)
            self.u += self.dt_msec * self.a * (self.b * self.v - self.u)
        except OverflowError:
            # Reset without producing a spike.
            self.v = self.c
            self.u = self.b * self.v

        self.output = 0.0
        if self.v > 30.0:
            # Output spike and reset.
            self.output = 1.0
            self.v = self.c
            self.u += self.d

    def reset(self):
        """Resets all state variables."""
        self.v = self.c
        self.u = self.b * self.v
        self.output = 0.0
        self.current = self.bias


class IzNetwork(object):
    def __init__(self, neurons, inputs, outputs, connections):
        self.neurons = neurons
        self.connections = []
        all_nodes = inputs + outputs + list(self.neurons.keys())
        for i, o, w in connections:
            self.connections.append((neurons[i], o, w))
            all_nodes += [i, o]

        self.inputs = inputs
        self.outputs = outputs
        max_node = max(all_nodes)
        self.currents = [0.0] * (1 + max_node)

    def set_inputs(self, inputs):
        assert len(inputs) == len(self.inputs)
        for i, v in zip(self.inputs, inputs):
            self.currents[i] = 0.0
            self.neurons[i].current = 0.0
            self.neurons[i].output = v

    def reset(self):
        # Reset all neurons.
        for i, n in self.neurons.items():
            n.reset()

    def advance(self):
        # Initialize all non-input neuron currents to the bias value.
        for i, n in self.neurons.items():
            if i not in self.inputs:
                self.currents[i] = n.bias

        # Add weight-adjusted output currents.
        for i, o, w in self.connections:
            self.currents[o] += i.output * w

        for i, n in self.neurons.items():
            if i not in self.inputs:
                n.current = self.currents[i]
                n.advance()

        return [self.neurons[i].output for i in self.outputs]


def create_phenotype(genome, a, b, c, d, time_step_msec=1.0):
    """ Receives a genome and returns its phenotype (a neural network) """

    neurons = {}
    inputs = []
    outputs = []
    for ng in genome.node_genes.values():
        # TODO: It seems like we should have a separate node gene implementation
        # that encodes more (all?) of the Izhikevitch model parameters.
        neurons[ng.ID] = Neuron(ng.bias, a, b, c, d, time_step_msec)
        if ng.type == 'INPUT':
            inputs.append(ng.ID)
        elif ng.type == 'OUTPUT':
            outputs.append(ng.ID)

    connections = []
    for cg in genome.conn_genes.values():
        if cg.enabled:
            connections.append((cg.in_node_id, cg.out_node_id, cg.weight))

    return IzNetwork(neurons, inputs, outputs, connections)
