import random

from neat import nn
from neat.config import Config
from neat.genome import DefaultGenome


def assert_almost_equal(x, y, tol):
    assert abs(x - y) < tol, "{!r} !~= {!r}".format(x, y)


def test_creates_cycle():
    assert nn.creates_cycle([(0, 1), (1, 2), (2, 3)], (0, 0))

    assert nn.creates_cycle([(0, 1), (1, 2), (2, 3)], (1, 0))
    assert not nn.creates_cycle([(0, 1), (1, 2), (2, 3)], (0, 1))

    assert nn.creates_cycle([(0, 1), (1, 2), (2, 3)], (2, 0))
    assert not nn.creates_cycle([(0, 1), (1, 2), (2, 3)], (0, 2))

    assert nn.creates_cycle([(0, 1), (1, 2), (2, 3)], (3, 0))
    assert not nn.creates_cycle([(0, 1), (1, 2), (2, 3)], (0, 3))

    assert nn.creates_cycle([(0, 2), (1, 3), (2, 3), (4, 2)], (3, 4))
    assert not nn.creates_cycle([(0, 2), (1, 3), (2, 3), (4, 2)], (4, 3))


def test_required_for_output():
    inputs = [0, 1]
    outputs = [2]
    connections = [(0, 2), (1, 2)]
    required = nn.required_for_output(inputs, outputs, connections)
    assert {2} == required

    inputs = [0, 1]
    outputs = [2]
    connections = [(0, 3), (1, 4), (3, 2), (4, 2)]
    required = nn.required_for_output(inputs, outputs, connections)
    assert {2, 3, 4} == required

    inputs = [0, 1]
    outputs = [3]
    connections = [(0, 2), (1, 2), (2, 3)]
    required = nn.required_for_output(inputs, outputs, connections)
    assert {2, 3} == required

    inputs = [0, 1]
    outputs = [4]
    connections = [(0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    required = nn.required_for_output(inputs, outputs, connections)
    assert {2, 3, 4} == required

    inputs = [0, 1]
    outputs = [4]
    connections = [(0, 2), (1, 3), (2, 3), (3, 4), (4, 2)]
    required = nn.required_for_output(inputs, outputs, connections)
    assert {2, 3, 4} == required

    inputs = [0, 1]
    outputs = [4]
    connections = [(0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5)]
    required = nn.required_for_output(inputs, outputs, connections)
    assert {2, 3, 4} == required


def test_fuzz_required():
    for _ in range(1000):
        n_hidden = random.randint(10, 100)
        n_in = random.randint(1, 10)
        n_out = random.randint(1, 10)
        nodes = list(set(random.randint(0, 1000) for _ in range(n_in + n_out + n_hidden)))
        random.shuffle(nodes)

        inputs = nodes[:n_in]
        outputs = nodes[n_in:n_in + n_out]
        connections = []
        for _ in range(n_hidden * 2):
            a = random.choice(nodes)
            b = random.choice(nodes)
            if a == b:
                continue
            if a in inputs and b in inputs:
                continue
            if a in outputs and b in outputs:
                continue
            connections.append((a, b))

        required = nn.required_for_output(inputs, outputs, connections)
        for o in outputs:
            assert o in required


def test_feed_forward_layers():
    inputs = [0, 1]
    outputs = [2]
    connections = [(0, 2), (1, 2)]
    layers = nn.feed_forward_layers(inputs, outputs, connections)
    assert [{2}] == layers

    inputs = [0, 1]
    outputs = [3]
    connections = [(0, 2), (1, 2), (2, 3)]
    layers = nn.feed_forward_layers(inputs, outputs, connections)
    assert [{2}, {3}] == layers

    inputs = [0, 1]
    outputs = [4]
    connections = [(0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    layers = nn.feed_forward_layers(inputs, outputs, connections)
    assert [{2}, {3}, {4}] == layers

    inputs = [0, 1, 2, 3]
    outputs = [11, 12, 13]
    connections = [(0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7),
                   (4, 8), (5, 8), (5, 9), (5, 10), (6, 10), (6, 7),
                   (8, 11), (8, 12), (8, 9), (9, 10), (7, 10),
                   (10, 12), (10, 13)]
    layers = nn.feed_forward_layers(inputs, outputs, connections)
    #print(layers)
    assert [{4, 5, 6}, {8, 7}, {9, 11}, {10}, {12, 13}] == layers

    inputs = [0, 1, 2, 3]
    outputs = [11, 12, 13]
    connections = [(0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7),
                   (4, 8), (5, 8), (5, 9), (5, 10), (6, 10), (6, 7),
                   (8, 11), (8, 12), (8, 9), (9, 10), (7, 10),
                   (10, 12), (10, 13),
                   (3, 14), (14, 15), (5, 16), (10, 16)]
    layers = nn.feed_forward_layers(inputs, outputs, connections)
    #print(layers)
    assert [{4, 5, 6}, {8, 7}, {9, 11}, {10}, {12, 13}] == layers


def test_fuzz_feed_forward_layers():
    for _ in range(1000):
        n_hidden = random.randint(10, 100)
        n_in = random.randint(1, 10)
        n_out = random.randint(1, 10)
        nodes = list(set(random.randint(0, 1000) for _ in range(n_in + n_out + n_hidden)))
        random.shuffle(nodes)

        inputs = nodes[:n_in]
        outputs = nodes[n_in:n_in + n_out]
        connections = []
        for _ in range(n_hidden * 2):
            a = random.choice(nodes)
            b = random.choice(nodes)
            if a == b:
                continue
            if a in inputs and b in inputs:
                continue
            if a in outputs and b in outputs:
                continue
            connections.append((a, b))

        nn.feed_forward_layers(inputs, outputs, connections)


def test_simple_nohidden():
    config = Config()
    config.genome_config.set_input_output_sizes(2, 1)
    g = DefaultGenome(0, config)
    g.add_node(0, 0.0, 1.0, 'sum', 'tanh')
    g.add_connection(-1, 0, 1.0, True)
    g.add_connection(-2, 0, -1.0, True)

    net = nn.create_feed_forward_phenotype(g, config)

    v00 = net.serial_activate([0.0, 0.0])
    assert_almost_equal(v00[0], 0.0, 1e-3)

    v01 = net.serial_activate([0.0, 1.0])
    assert_almost_equal(v01[0], -0.76159, 1e-3)

    v10 = net.serial_activate([1.0, 0.0])
    assert_almost_equal(v10[0], 0.76159, 1e-3)

    v11 = net.serial_activate([1.0, 1.0])
    assert_almost_equal(v11[0], 0.0, 1e-3)


def test_simple_hidden():
    config = Config()
    config.genome_config.set_input_output_sizes(2, 1)
    g = DefaultGenome(0, config)

    g.add_node(0, 0.0, 1.0, 'sum', 'identity')
    g.add_node(1, -0.5, 5.0, 'sum', 'sigmoid')
    g.add_node(2, -1.5, 5.0, 'sum', 'sigmoid')
    g.add_connection(-1, 1, 1.0, True)
    g.add_connection(-2, 2, 1.0, True)
    g.add_connection(1, 0, 1.0, True)
    g.add_connection(2, 0, -1.0, True)
    net = nn.create_feed_forward_phenotype(g, config)

    v00 = net.serial_activate([0.0, 0.0])
    assert_almost_equal(v00[0], 0.195115, 1e-3)

    v01 = net.serial_activate([0.0, 1.0])
    assert_almost_equal(v01[0], -0.593147, 1e-3)

    v10 = net.serial_activate([1.0, 0.0])
    assert_almost_equal(v10[0], 0.806587, 1e-3)

    v11 = net.serial_activate([1.0, 1.0])
    assert_almost_equal(v11[0], 0.018325, 1e-3)


if __name__ == '__main__':
    test_creates_cycle()
    test_required_for_output()
    test_fuzz_required()
    test_feed_forward_layers()
    test_fuzz_feed_forward_layers()
    test_simple_nohidden()
    test_simple_hidden()
