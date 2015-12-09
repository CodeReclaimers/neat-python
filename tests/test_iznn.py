from neat.iznn import Neuron, IzNetwork


def test_basic():
    n = Neuron(10, 0.02, 0.2, -65.0, 8.0)
    spike_train = []
    for i in range(1000):
        spike_train.append(n.v)
        n.advance()


def test_network():
    neurons = {0: Neuron(0, 0.02, 0.2, -65.0, 8.0),
               1: Neuron(0, 0.02, 0.2, -65.0, 8.0),
               2: Neuron(0, 0.02, 0.2, -65.0, 8.0)}
    inputs = [0, 1]
    outputs = [2]
    connections = [(0, 2, 0.123), (1, 2, 0.234)]

    net = IzNetwork(neurons, inputs, outputs, connections)
    net.set_inputs([1.0, 0.0])
    net.advance()
    net.advance()
