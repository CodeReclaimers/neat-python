from neat import nn
from neat.genome import FFGenome
from neat.genes import NodeGene, ConnectionGene


def test_find_feed_forward_layers():
    inputs = [0, 1]
    connections = [(0, 2), (1, 2)]
    layers = nn.find_feed_forward_layers(inputs, connections)
    assert [{2}] == layers

    inputs = [0, 1]
    connections = [(0, 2), (1, 2), (2, 3)]
    layers = nn.find_feed_forward_layers(inputs, connections)
    assert [{2}, {3}] == layers

    inputs = [0, 1]
    connections = [(0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    layers = nn.find_feed_forward_layers(inputs, connections)
    assert [{2}, {3}, {4}] == layers

    inputs = [0, 1, 2, 3]
    connections = [(0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7),
                   (4, 8), (5, 8), (5, 9), (5, 10), (6, 10), (6, 7),
                   (8, 11), (8, 12), (8, 9), (9, 10), (7, 10),
                   (10, 12), (10, 13)]
    layers = nn.find_feed_forward_layers(inputs, connections)
    assert [{4, 5, 6}, {8, 7}, {9, 11}, {10}, {12, 13}] == layers


def test_simple():
    g = FFGenome(0)
    g.inputs[0] = NodeGene(0, 0.0, 1.0, 'sum', 'identity')
    g.inputs[1] = NodeGene(1, 0.0, 1.0, 'sum', 'identity')
    g.outputs[2] = NodeGene(2, 0.0, 5.0, 'sum', 'sigmoid')
    g.connections[(0,2)] = ConnectionGene(0, 2, 1.0, True)
    g.connections[(1,2)] = ConnectionGene(1, 2, -1.0, True)
    net = nn.create_feed_forward_phenotype(g)

    net.serial_activate([0.0, 0.0])
    net.serial_activate([0.0, 1.0])
    net.serial_activate([1.0, 0.0])
    net.serial_activate([1.0, 1.0])


if __name__ == '__main__':
    test_simple()
