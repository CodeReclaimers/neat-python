from neat import activations
from neat.nn import RecurrentNetwork


def assert_almost_equal(x, y, tol):
    assert abs(x - y) < tol, "{!r} !~= {!r}".format(x, y)


def test_unconnected():
    # Unconnected network with no inputs and one output neuron.
    node_evals = [(0, activations.sigmoid_activation, sum, 0.0, 1.0, [])]
    r = RecurrentNetwork([], [0], node_evals)

    assert r.active == 0
    assert len(r.values) == 2
    assert len(r.values[0]) == 1
    assert len(r.values[1]) == 1

    result = r.activate([])

    assert r.active == 1
    assert_almost_equal(r.values[1][0], 0.5, 0.001)
    assert result[0] == r.values[1][0]

    result = r.activate([])

    assert r.active == 0
    assert_almost_equal(r.values[0][0], 0.5, 0.001)
    assert result[0] == r.values[0][0]


def test_basic():
    # Very simple network with one connection of weight one to a single sigmoid output node.
    node_evals = [(0, activations.sigmoid_activation, sum, 0.0, 1.0, [(-1, 1.0)])]
    r = RecurrentNetwork([-1], [0], node_evals)

    assert r.active == 0
    assert len(r.values) == 2
    assert len(r.values[0]) == 2
    assert len(r.values[1]) == 2

    result = r.activate([0.2])

    assert r.active == 1
    assert r.values[1][-1] == 0.2
    assert_almost_equal(r.values[1][0], 0.731, 0.001)
    assert result[0] == r.values[1][0]

    result = r.activate([0.4])

    assert r.active == 0
    assert r.values[0][-1] == 0.4
    assert_almost_equal(r.values[0][0], 0.881, 0.001)
    assert result[0] == r.values[0][0]


if __name__ == '__main__':
    test_unconnected()
    test_basic()
