import random
from neat.graphs import creates_cycle, required_for_output, feed_forward_layers


def assert_almost_equal(x, y, tol):
    assert abs(x - y) < tol, "{!r} !~= {!r}".format(x, y)


def test_creates_cycle():
    assert creates_cycle([(0, 1), (1, 2), (2, 3)], (0, 0))

    assert creates_cycle([(0, 1), (1, 2), (2, 3)], (1, 0))
    assert not creates_cycle([(0, 1), (1, 2), (2, 3)], (0, 1))

    assert creates_cycle([(0, 1), (1, 2), (2, 3)], (2, 0))
    assert not creates_cycle([(0, 1), (1, 2), (2, 3)], (0, 2))

    assert creates_cycle([(0, 1), (1, 2), (2, 3)], (3, 0))
    assert not creates_cycle([(0, 1), (1, 2), (2, 3)], (0, 3))

    assert creates_cycle([(0, 2), (1, 3), (2, 3), (4, 2)], (3, 4))
    assert not creates_cycle([(0, 2), (1, 3), (2, 3), (4, 2)], (4, 3))


def test_required_for_output():
    inputs = [0, 1]
    outputs = [2]
    connections = [(0, 2), (1, 2)]
    required = required_for_output(inputs, outputs, connections)
    assert {2} == required

    inputs = [0, 1]
    outputs = [2]
    connections = [(0, 3), (1, 4), (3, 2), (4, 2)]
    required = required_for_output(inputs, outputs, connections)
    assert {2, 3, 4} == required

    inputs = [0, 1]
    outputs = [3]
    connections = [(0, 2), (1, 2), (2, 3)]
    required = required_for_output(inputs, outputs, connections)
    assert {2, 3} == required

    inputs = [0, 1]
    outputs = [4]
    connections = [(0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    required = required_for_output(inputs, outputs, connections)
    assert {2, 3, 4} == required

    inputs = [0, 1]
    outputs = [4]
    connections = [(0, 2), (1, 3), (2, 3), (3, 4), (4, 2)]
    required = required_for_output(inputs, outputs, connections)
    assert {2, 3, 4} == required

    inputs = [0, 1]
    outputs = [4]
    connections = [(0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (2, 5)]
    required = required_for_output(inputs, outputs, connections)
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

        required = required_for_output(inputs, outputs, connections)
        for o in outputs:
            assert o in required


def test_feed_forward_layers():
    inputs = [0, 1]
    outputs = [2]
    connections = [(0, 2), (1, 2)]
    layers = feed_forward_layers(inputs, outputs, connections)
    assert [{2}] == layers

    inputs = [0, 1]
    outputs = [3]
    connections = [(0, 2), (1, 2), (2, 3)]
    layers = feed_forward_layers(inputs, outputs, connections)
    assert [{2}, {3}] == layers

    inputs = [0, 1]
    outputs = [4]
    connections = [(0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    layers = feed_forward_layers(inputs, outputs, connections)
    assert [{2}, {3}, {4}] == layers

    inputs = [0, 1, 2, 3]
    outputs = [11, 12, 13]
    connections = [(0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7),
                   (4, 8), (5, 8), (5, 9), (5, 10), (6, 10), (6, 7),
                   (8, 11), (8, 12), (8, 9), (9, 10), (7, 10),
                   (10, 12), (10, 13)]
    layers = feed_forward_layers(inputs, outputs, connections)
    assert [{4, 5, 6}, {8, 7}, {9, 11}, {10}, {12, 13}] == layers

    inputs = [0, 1, 2, 3]
    outputs = [11, 12, 13]
    connections = [(0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7),
                   (4, 8), (5, 8), (5, 9), (5, 10), (6, 10), (6, 7),
                   (8, 11), (8, 12), (8, 9), (9, 10), (7, 10),
                   (10, 12), (10, 13),
                   (3, 14), (14, 15), (5, 16), (10, 16)]
    layers = feed_forward_layers(inputs, outputs, connections)
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

        feed_forward_layers(inputs, outputs, connections)


if __name__ == '__main__':
    test_creates_cycle()
    test_required_for_output()
    test_fuzz_required()
    test_feed_forward_layers()
    test_fuzz_feed_forward_layers()
