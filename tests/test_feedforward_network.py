from neat import activations
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


# TODO: Update this test for the current implementation.
# def test_simple_nohidden():
#     config_params = {
#         'num_inputs':2,
#         'num_outputs':1,
#         'num_hidden':0,
#         'feed_forward':True,
#         'compatibility_threshold':3.0,
#         'excess_coefficient':1.0,
#         'disjoint_coefficient':1.0,
#         'compatibility_weight_coefficient':1.0,
#         'conn_add_prob':0.5,
#         'conn_delete_prob':0.05,
#         'node_add_prob':0.1,
#         'node_delete_prob':0.05}
#     config = DefaultGenomeConfig(config_params)
#     config.genome_config.set_input_output_sizes(2, 1)
#     g = DefaultGenome(0, config)
#     g.add_node(0, 0.0, 1.0, 'sum', 'tanh')
#     g.add_connection(-1, 0, 1.0, True)
#     g.add_connection(-2, 0, -1.0, True)
#
#     net = nn.create_feed_forward_phenotype(g, config)
#
#     v00 = net.serial_activate([0.0, 0.0])
#     assert_almost_equal(v00[0], 0.0, 1e-3)
#
#     v01 = net.serial_activate([0.0, 1.0])
#     assert_almost_equal(v01[0], -0.76159, 1e-3)
#
#     v10 = net.serial_activate([1.0, 0.0])
#     assert_almost_equal(v10[0], 0.76159, 1e-3)
#
#     v11 = net.serial_activate([1.0, 1.0])
#     assert_almost_equal(v11[0], 0.0, 1e-3)


# TODO: Update this test for the current implementation.
# def test_simple_hidden():
#     config = Config()
#     config.genome_config.set_input_output_sizes(2, 1)
#     g = DefaultGenome(0, config)
#
#     g.add_node(0, 0.0, 1.0, 'sum', 'identity')
#     g.add_node(1, -0.5, 5.0, 'sum', 'sigmoid')
#     g.add_node(2, -1.5, 5.0, 'sum', 'sigmoid')
#     g.add_connection(-1, 1, 1.0, True)
#     g.add_connection(-2, 2, 1.0, True)
#     g.add_connection(1, 0, 1.0, True)
#     g.add_connection(2, 0, -1.0, True)
#     net = nn.create_feed_forward_phenotype(g, config)
#
#     v00 = net.serial_activate([0.0, 0.0])
#     assert_almost_equal(v00[0], 0.195115, 1e-3)
#
#     v01 = net.serial_activate([0.0, 1.0])
#     assert_almost_equal(v01[0], -0.593147, 1e-3)
#
#     v10 = net.serial_activate([1.0, 0.0])
#     assert_almost_equal(v10[0], 0.806587, 1e-3)
#
#     v11 = net.serial_activate([1.0, 1.0])
#     assert_almost_equal(v11[0], 0.018325, 1e-3)


if __name__ == '__main__':
    test_unconnected()
    test_basic()
