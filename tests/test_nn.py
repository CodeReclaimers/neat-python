import math
import os
import random

import neat
import pytest
from neat import activations
from neat.genes import DefaultNodeGene, DefaultConnectionGene


def test_orphaned_node_network():
    """Test that networks with orphaned nodes (no incoming connections) work correctly."""
    # Load configuration
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # Create a genome with an orphaned hidden node
    genome = neat.DefaultGenome(1)
    genome.fitness = None
    
    # Manually create nodes:
    # - Output node 0
    # - Hidden node 1 (orphaned - no incoming connections)
    node_0 = DefaultNodeGene(0)
    node_0.bias = 0.5
    node_0.response = 1.0
    node_0.activation = 'sigmoid'
    node_0.aggregation = 'sum'
    genome.nodes[0] = node_0
    
    node_1 = DefaultNodeGene(1)
    node_1.bias = 2.0  # This bias value should affect the output
    node_1.response = 1.0
    node_1.activation = 'sigmoid'
    node_1.aggregation = 'sum'
    genome.nodes[1] = node_1
    
    # Manually create connections:
    # - Node 1 (orphaned) connects to output node 0
    # - Note: node 1 has no incoming connections (it's orphaned)
    conn_key = (1, 0)
    conn = DefaultConnectionGene(conn_key, innovation=0)  # Innovation number required
    conn.weight = 1.0
    conn.enabled = True
    genome.connections[conn_key] = conn
    
    # Create the feed-forward network
    # This should not crash despite node 1 being orphaned
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Activate the network
    # The orphaned node 1 should contribute its activation(bias) to the output
    output = net.activate([0.0, 0.0])  # Two inputs (as per test_configuration)
    
    # Verify the network produces output
    assert len(output) == 1
    assert output[0] is not None
    
    # Verify the output is affected by the orphaned node's bias
    # NOTE: neat-python's sigmoid_activation multiplies input by 5.0 before applying sigmoid!
    # Node 1 with bias=2.0, no inputs -> sigmoid(5.0 * (2.0 + 1.0 * 0)) = sigmoid(10.0) ≈ 0.9999
    # Node 1 output (0.9999) * weight (1.0) goes to node 0
    # Node 0: sigmoid(5.0 * (0.5 + 1.0 * 0.9999)) = sigmoid(5.0 * 1.4999) = sigmoid(7.5) ≈ 0.9994
    
    # Calculate using neat-python's sigmoid_activation (with 5.0 scaling)
    def neat_sigmoid(z):
        z = max(-60.0, min(60.0, 5.0 * z))
        return 1.0 / (1.0 + math.exp(-z))
    
    expected_node_1_output = neat_sigmoid(2.0)  # bias only, no inputs
    expected_output = neat_sigmoid(0.5 + expected_node_1_output)  # bias + weighted input from node 1
    
    assert abs(output[0] - expected_output) < 0.001, f"Expected {expected_output:.6f}, got {output[0]:.6f}"


def test_feedforward_input_length_mismatch_raises():
    """FeedForwardNetwork.activate should enforce the expected input length."""
    node_evals = [(0, activations.sigmoid_activation, sum, 0.0, 1.0, [(-1, 1.0)])]
    net = neat.nn.FeedForwardNetwork([-1], [0], node_evals)

    # Too few inputs.
    with pytest.raises(RuntimeError, match="Expected 1 inputs, got 0"):
        net.activate([])

    # Too many inputs.
    with pytest.raises(RuntimeError, match="Expected 1 inputs, got 2"):
        net.activate([0.1, 0.2])


def test_recurrent_input_length_mismatch_raises():
    """RecurrentNetwork.activate should enforce the expected input length."""
    node_evals = [(0, activations.sigmoid_activation, sum, 0.0, 1.0, [(-1, 1.0)])]
    net = neat.nn.RecurrentNetwork([-1], [0], node_evals)

    # Too few inputs.
    with pytest.raises(RuntimeError, match="Expected 1 inputs, got 0"):
        net.activate([])

    # Too many inputs.
    with pytest.raises(RuntimeError, match="Expected 1 inputs, got 2"):
        net.activate([0.1, 0.2])


def _create_simple_recurrent_network():
    """Small recurrent net with one input and a self-connection on the output node."""
    node_evals = [
        (0, activations.sigmoid_activation, sum, 0.0, 1.0, [(-1, 1.0), (0, 1.0)]),
    ]
    return neat.nn.RecurrentNetwork([-1], [0], node_evals)


def test_recurrent_reset_zeros_state_and_resets_active():
    """reset() should zero internal state buffers and reset the active index."""
    net = _create_simple_recurrent_network()

    # Drive the network so that internal state changes.
    net.activate([0.5])
    net.activate([0.3])
    assert any(any(value != 0.0 for value in layer.values()) for layer in net.values)

    net.reset()
    assert net.active == 0
    assert all(all(value == 0.0 for value in layer.values()) for layer in net.values)


def test_recurrent_create_from_genome_prunes_unused_nodes():
    """RecurrentNetwork.create should respect required_for_output and prune unused nodes."""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    genome = neat.DefaultGenome(1)

    # Manually create nodes: one output (0), one used hidden (1), one unused hidden (2).
    node0 = DefaultNodeGene(0)
    node0.bias = 0.1
    node0.response = 1.0
    node0.activation = 'sigmoid'
    node0.aggregation = 'sum'

    node1 = DefaultNodeGene(1)
    node1.bias = -0.2
    node1.response = 1.0
    node1.activation = 'sigmoid'
    node1.aggregation = 'sum'

    node2 = DefaultNodeGene(2)
    node2.bias = 1.5
    node2.response = 1.0
    node2.activation = 'sigmoid'
    node2.aggregation = 'sum'

    genome.nodes[0] = node0
    genome.nodes[1] = node1
    genome.nodes[2] = node2

    # Connections: input -1 -> hidden 1, hidden 1 -> output 0, and an unused input -2 -> hidden 2.
    conn1_key = (-1, 1)
    conn1 = DefaultConnectionGene(conn1_key, innovation=0)
    conn1.weight = 0.5
    conn1.enabled = True

    conn2_key = (1, 0)
    conn2 = DefaultConnectionGene(conn2_key, innovation=1)
    conn2.weight = 1.5
    conn2.enabled = True

    conn3_key = (-2, 2)
    conn3 = DefaultConnectionGene(conn3_key, innovation=2)
    conn3.weight = -0.7
    conn3.enabled = True

    genome.connections[conn1_key] = conn1
    genome.connections[conn2_key] = conn2
    genome.connections[conn3_key] = conn3

    net = neat.nn.RecurrentNetwork.create(genome, config)
    genome_config = config.genome_config

    # Input and output node lists should come from the genome config.
    assert net.input_nodes == genome_config.input_keys
    assert net.output_nodes == genome_config.output_keys

    # Only nodes that are actually required for the outputs should have node_evals.
    node_eval_map = {
        node_key: (activation, aggregation, bias, response, inputs)
        for node_key, activation, aggregation, bias, response, inputs in net.node_evals
    }
    assert set(node_eval_map.keys()) == {0, 1}

    hidden_eval = node_eval_map[1]
    assert hidden_eval[2] == node1.bias
    assert hidden_eval[3] == node1.response
    assert hidden_eval[0] is genome_config.activation_defs.get(node1.activation)
    assert hidden_eval[1] is genome_config.aggregation_function_defs.get(node1.aggregation)
    assert hidden_eval[4] == [(-1, 0.5)]

    output_eval = node_eval_map[0]
    assert output_eval[2] == node0.bias
    assert output_eval[3] == node0.response
    assert output_eval[0] is genome_config.activation_defs.get(node0.activation)
    assert output_eval[1] is genome_config.aggregation_function_defs.get(node0.aggregation)
    assert output_eval[4] == [(1, 1.5)]


def _build_simple_feedforward_genome():
    """Small genome-built feedforward net: 2 inputs -> 1 identity output, no hidden layer."""
    genome = neat.DefaultGenome(2)

    node0 = DefaultNodeGene(0)
    node0.bias = 0.0
    node0.response = 1.0
    node0.activation = 'identity'
    node0.aggregation = 'sum'
    genome.nodes[0] = node0

    conn_keys = [(-1, 0), (-2, 0)]
    for innovation, key in enumerate(conn_keys):
        cg = DefaultConnectionGene(key, innovation=innovation)
        cg.weight = 0.0
        cg.enabled = True
        genome.connections[key] = cg

    return genome


def test_feedforward_create_unique_value_sets_all_weights_and_affects_output():
    """FeedForwardNetwork.create(unique_value=...) should override all connection weights."""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    genome = _build_simple_feedforward_genome()
    unique_weight = 0.37

    net = neat.nn.FeedForwardNetwork.create(genome, config, unique_value=unique_weight)

    # All connection genes should have the same weight.
    assert {cg.weight for cg in genome.connections.values()} == {unique_weight}

    # With identity activation and sum aggregation, output is unique_weight * sum(inputs).
    output = net.activate([1.0, 1.0])
    assert abs(output[0] - unique_weight * 2.0) < 1e-9


def test_feedforward_create_random_values_assigns_weights_in_range():
    """FeedForwardNetwork.create(random_values=True) should re-sample connection weights."""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    genome = _build_simple_feedforward_genome()

    # Temporarily seed the RNG used by neat.nn.feed_forward to keep this test stable.
    state = random.getstate()
    try:
        random.seed(42)
        net = neat.nn.FeedForwardNetwork.create(genome, config, random_values=True)
    finally:
        random.setstate(state)

    weights = [cg.weight for cg in genome.connections.values()]
    assert all(-1.0 <= w <= 1.0 for w in weights)
    assert any(w != 0.0 for w in weights)

    # Sanity check: network is still callable and produces a single output value.
    result = net.activate([0.5, -0.5])
    assert len(result) == 1


if __name__ == '__main__':
    test_orphaned_node_network()
