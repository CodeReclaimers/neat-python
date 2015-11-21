from neat.nn import find_feed_forward_layers


def test_simple():
    inputs = [0, 1]
    connections = [(0, 2), (1, 2)]
    layers = find_feed_forward_layers(inputs, connections)
    assert [{2}] == layers

    inputs = [0, 1]
    connections = [(0, 2), (1, 2), (2, 3)]
    layers = find_feed_forward_layers(inputs, connections)
    assert [{2}, {3}] == layers

    inputs = [0, 1]
    connections = [(0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    layers = find_feed_forward_layers(inputs, connections)
    assert [{2}, {3}, {4}] == layers

    inputs = [0, 1, 2, 3]
    connections = [(0, 4), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7),
                   (4, 8), (5, 8), (5, 9), (5, 10), (6, 10), (6, 7),
                   (8, 11), (8, 12), (8, 9), (9, 10), (7, 10),
                   (10, 12), (10, 13)]
    layers = find_feed_forward_layers(inputs, connections)
    assert [{4, 5, 6}, {8, 7}, {9, 11}, {10}, {12, 13}] == layers


if __name__ == '__main__':
    test_simple()
