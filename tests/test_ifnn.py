from neat.ifnn import IFNeuron, IFNetwork


def test_basic():
    n = IFNeuron(20)
    spike_train = []
    for i in range(1000):
        spike_train.append(n.potential)
        n.advance()


def test_network():
    neurons = {0: IFNeuron(5),
               1: IFNeuron(10),
               2: IFNeuron(20)}
    inputs = [0, 1]
    outputs = [2]
    connections = [(0, 2, 0.123), (1, 2, 0.234)]

    net = IFNetwork(neurons, inputs, outputs, connections)
    #net.advance([1.0, 0.0])
    #net.advance([1.0, 0.0])
