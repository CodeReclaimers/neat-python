from neat.nn.nn_pure import FeedForward, Neuron, Network


def test_feed_forward():
    nn = FeedForward([2, 10, 3], activation_type='exp')
    nn.sactivate([1, 1])
    nn.pactivate([1,1])


def test_manual_network():
    neurons = [Neuron('INPUT', 1), Neuron('HIDDEN', 2), Neuron('OUTPUT', 3)]
    connections = [(1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.5)]

    net = Network(neurons, connections, 1)
    net.sactivate([0.04])
    net.pactivate([0.04])
    repr(net)
