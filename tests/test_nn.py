from neat import ctrnn

# TODO: These tests are just smoke tests to make sure nothing has become badly broken.  Expand
# to include more detailed tests of actual functionality.


def create_simple():
    neurons = [ctrnn.Neuron('INPUT', 1, 0.0, 5.0, 'sigmoid'),
               ctrnn.Neuron('HIDDEN', 2, 0.0, 5.0, 'sigmoid'),
               ctrnn.Neuron('OUTPUT', 3, 0.0, 5.0, 'sigmoid')]
    connections = [(1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.5)]

    return ctrnn.Network(neurons, connections, 1)


def test_manual_network():
    net = create_simple()
    net.serial_activate([0.04])
    net.parallel_activate([0.04])
    repr(net)
