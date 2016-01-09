from __future__ import print_function

from neat.ctrnn import CTNeuron, Neuron, Network

# TODO: These tests are just smoke tests to make sure nothing has become badly broken.  Expand
# to include more detailed tests of actual functionality.

def test_basic():
    # create two output neurons (they won't receive any external inputs)
    n1 = CTNeuron('OUTPUT', 1, -2.75, 1.0, 'sigmoid', 0.5)
    n2 = CTNeuron('OUTPUT', 2, -1.75, 1.0, 'sigmoid', 0.5)
    n1.set_init_state(-0.084000643)
    n2.set_init_state(-0.408035109)

    neurons_list = [n1, n2]
    # create some synapses
    conn_list = [(1, 1, 4.5), (1, 2, -1.0), (2, 1, 1.0), (2, 2, 4.5)]
    # create the network
    net = Network(neurons_list, conn_list, 0)
    # activates the network
    print("{0:.7f} {1:.7f}".format(n1.output, n2.output))
    outputs = []
    for i in range(1000):
        output = net.parallel_activate()
        outputs.append(output)
        print("{0:.7f} {1:.7f}".format(output[0], output[1]))


def create_simple():
    neurons = [Neuron('INPUT', 1, 0.0, 5.0, 'sigmoid'),
               Neuron('HIDDEN', 2, 0.0, 5.0, 'sigmoid'),
               Neuron('OUTPUT', 3, 0.0, 5.0, 'sigmoid')]
    connections = [(1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.5)]

    return Network(neurons, connections, 1)


def test_manual_network():
    net = create_simple()
    net.serial_activate([0.04])
    net.parallel_activate([0.04])
    repr(net)
