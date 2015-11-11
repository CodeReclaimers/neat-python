from neat.iznn import Neuron


def test_basic():
    n = Neuron(10)
    spike_train = []
    for i in range(1000):
        spike_train.append(n.potential)
        n.advance()
