import os

import neat
from neat import activations
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.nn import FeedForwardNetwork


def assert_almost_equal(x, y, tol):
    assert abs(x - y) < tol, "{!r} !~= {!r}".format(x, y)


def test_unconnected():
    # Unconnected network with no inputs and one output neuron.
    node_evals = [(0, activations.sigmoid_activation, sum, 0.0, 1.0, [])]
    r = FeedForwardNetwork([], [0], node_evals)

    assert r.values[0] == 0.0

    result = r.activate([])

    assert_almost_equal(r.values[0], 0.5, 0.001)
    assert result[0] == r.values[0]

    result = r.activate([])

    assert_almost_equal(r.values[0], 0.5, 0.001)
    assert result[0] == r.values[0]


def test_basic():
    # Very simple network with one connection of weight one to a single sigmoid output node.
    node_evals = [(0, activations.sigmoid_activation, sum, 0.0, 1.0, [(-1, 1.0)])]
    r = FeedForwardNetwork([-1], [0], node_evals)

    assert r.values[0] == 0.0

    result = r.activate([0.2])

    assert r.values[-1] == 0.2
    assert_almost_equal(r.values[0], 0.731, 0.001)
    assert result[0] == r.values[0]

    result = r.activate([0.4])

    assert r.values[-1] == 0.4
    assert_almost_equal(r.values[0], 0.881, 0.001)
    assert result[0] == r.values[0]


def _create_simple_nohidden_network():
    """Small genome-built feedforward net: 2 inputs -> 1 tanh output, no hidden layer."""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "test_configuration")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    genome = neat.DefaultGenome(0)

    # Single output node 0 with tanh activation and sum aggregation.
    node0 = DefaultNodeGene(0)
    node0.bias = 0.0
    node0.response = 1.0
    node0.activation = "tanh"
    node0.aggregation = "sum"
    genome.nodes[0] = node0

    # Connections: input -1 -> 0 (weight 1.0), input -2 -> 0 (weight -1.0).
    conn1_key = (-1, 0)
    conn1 = DefaultConnectionGene(conn1_key, innovation=0)
    conn1.weight = 1.0
    conn1.enabled = True

    conn2_key = (-2, 0)
    conn2 = DefaultConnectionGene(conn2_key, innovation=1)
    conn2.weight = -1.0
    conn2.enabled = True

    genome.connections[conn1_key] = conn1
    genome.connections[conn2_key] = conn2

    return FeedForwardNetwork.create(genome, config)


def test_simple_nohidden_from_genome():
    """FeedForwardNetwork.create builds the expected simple no-hidden network."""
    net = _create_simple_nohidden_network()

    v00 = net.activate([0.0, 0.0])
    assert_almost_equal(v00[0], 0.0, 1e-6)

    v01 = net.activate([0.0, 1.0])
    assert_almost_equal(v01[0], -0.9866142981514303, 1e-6)

    v10 = net.activate([1.0, 0.0])
    assert_almost_equal(v10[0], 0.9866142981514303, 1e-6)

    v11 = net.activate([1.0, 1.0])
    assert_almost_equal(v11[0], 0.0, 1e-6)


def _create_simple_hidden_network():
    """Small genome-built feedforward net: 2 inputs -> 2 sigmoid hidden -> 1 identity output."""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "test_configuration")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    genome = neat.DefaultGenome(0)

    # Output node 0 (identity), hidden nodes 1 and 2 (sigmoid).
    node0 = DefaultNodeGene(0)
    node0.bias = 0.0
    node0.response = 1.0
    node0.activation = "identity"
    node0.aggregation = "sum"

    node1 = DefaultNodeGene(1)
    node1.bias = -0.5
    node1.response = 5.0
    node1.activation = "sigmoid"
    node1.aggregation = "sum"

    node2 = DefaultNodeGene(2)
    node2.bias = -1.5
    node2.response = 5.0
    node2.activation = "sigmoid"
    node2.aggregation = "sum"

    genome.nodes[0] = node0
    genome.nodes[1] = node1
    genome.nodes[2] = node2

    # Connections: -1 -> 1, -2 -> 2, and hidden 1/2 to output 0 (second with weight -1.0).
    connections = [
        ((-1, 1), 1.0),
        ((-2, 2), 1.0),
        ((1, 0), 1.0),
        ((2, 0), -1.0),
    ]

    for innovation, (key, weight) in enumerate(connections):
        cg = DefaultConnectionGene(key, innovation=innovation)
        cg.weight = weight
        cg.enabled = True
        genome.connections[key] = cg

    return FeedForwardNetwork.create(genome, config)


def test_simple_hidden_from_genome():
    """FeedForwardNetwork.create builds a simple hidden-layer network with expected behavior."""
    net = _create_simple_hidden_network()

    v00 = net.activate([0.0, 0.0])
    assert_almost_equal(v00[0], 0.07530540138431994, 1e-6)

    v01 = net.activate([0.0, 1.0])
    assert_almost_equal(v01[0], -0.9241417948687655, 1e-6)

    v10 = net.activate([1.0, 0.0])
    assert_almost_equal(v10[0], 0.9994472211938866, 1e-6)

    v11 = net.activate([1.0, 1.0])
    assert_almost_equal(v11[0], 2.4940801202077978e-08, 1e-6)


if __name__ == '__main__':
    test_unconnected()
    test_basic()
    test_simple_nohidden_from_genome()
    test_simple_hidden_from_genome()
