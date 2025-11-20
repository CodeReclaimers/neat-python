import os

import neat
import pytest
from neat.activations import sigmoid_activation
from neat.genes import DefaultConnectionGene, DefaultNodeGene


def _create_two_neuron_ctrnn():
    """Create the 2-neuron autonomous CTRNN used in the demo-ctrnn example."""
    # Fully-connected 2-neuron network with no external inputs.
    node1_inputs = [(1, 0.9), (2, 0.2)]
    node2_inputs = [(1, -0.2), (2, 0.9)]

    node_evals = {
        1: neat.ctrnn.CTRNNNodeEval(0.01, sigmoid_activation, sum, -2.75 / 5.0, 1.0, node1_inputs),
        2: neat.ctrnn.CTRNNNodeEval(0.01, sigmoid_activation, sum, -1.75 / 5.0, 1.0, node2_inputs),
    }

    net = neat.ctrnn.CTRNN([], [1, 2], node_evals)

    # Start both neurons from 0.0, matching the example script.
    net.set_node_value(1, 0.0)
    net.set_node_value(2, 0.0)

    return net


def test_basic_two_neuron_dynamics():
    """Basic numerical behavior test for a hand-constructed 2-neuron CTRNN."""
    net = _create_two_neuron_ctrnn()

    outputs = []
    num_steps = 1250
    dt = 0.002

    for _ in range(num_steps):
        output = net.advance([], dt, dt)
        outputs.append(output)

    # Total simulated time should be close to 2.5 seconds (1250 * 0.002).
    assert abs(net.time_seconds - 2.5) < 1e-9

    # All outputs should remain in [0, 1] due to sigmoid activation.
    for o in outputs:
        assert 0.0 <= o[0] <= 1.0
        assert 0.0 <= o[1] <= 1.0

    # Check specific reference values at selected timesteps to guard against
    # regressions in the CTRNN integration behavior.
    reference = {
        0: (0.0, 0.0),
        100: (0.3746852284, 0.8115273872),
        500: (0.5634208426, 0.1952743492),
        1149: (0.4092483263, 0.8024978195),
        1249: (0.7531862678, 0.3112381247),
    }
    tol = 1e-6
    for idx, (exp0, exp1) in reference.items():
        o = outputs[idx]
        assert abs(o[0] - exp0) < tol
        assert abs(o[1] - exp1) < tol


def test_reset_and_deterministic_trajectory():
    """CTRNN.reset should zero state and trajectories should be deterministic."""
    net = _create_two_neuron_ctrnn()

    def run(num_steps):
        seq = []
        for _ in range(num_steps):
            seq.append(net.advance([], 0.002, 0.002))
        return seq

    first_outputs = run(200)

    # State and time should have advanced.
    assert net.time_seconds > 0.0
    assert any(any(abs(v) > 0.0 for v in layer.values()) for layer in net.values)

    # Reset should restore time and all stored values to zero.
    net.reset()
    assert net.time_seconds == 0.0
    assert all(all(value == 0.0 for value in layer.values()) for layer in net.values)

    second_outputs = run(200)

    # Trajectories from the same initial conditions should match.
    assert len(first_outputs) == len(second_outputs)
    for o1, o2 in zip(first_outputs, second_outputs):
        for v1, v2 in zip(o1, o2):
            assert abs(v1 - v2) < 1e-12


def test_advance_input_validation():
    """advance should enforce input length and raise RuntimeError on mismatch."""
    # Simple CTRNN with a single input node feeding a single output node.
    node_inputs = [(0, 1.0)]
    node_evals = {
        1: neat.ctrnn.CTRNNNodeEval(1.0, sigmoid_activation, sum, 0.0, 1.0, node_inputs),
    }
    net = neat.ctrnn.CTRNN([0], [1], node_evals)

    # Sanity check: correct-length input works.
    net.advance([0.5], 0.1, 0.1)

    # Too few inputs.
    with pytest.raises(RuntimeError, match="Expected 1 inputs, got 0"):
        net.advance([], 0.1, 0.1)

    # Too many inputs.
    with pytest.raises(RuntimeError, match="Expected 1 inputs, got 2"):
        net.advance([0.1, 0.2], 0.1, 0.1)


def test_ctrnn_create_from_genome_prunes_and_builds_expected_structure():
    """CTRNN.create should respect required_for_output and build correct node_evals."""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "test_configuration")
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
    node0.activation = "sigmoid"
    node0.aggregation = "sum"

    node1 = DefaultNodeGene(1)
    node1.bias = -0.2
    node1.response = 1.0
    node1.activation = "sigmoid"
    node1.aggregation = "sum"

    node2 = DefaultNodeGene(2)
    node2.bias = 1.5
    node2.response = 1.0
    node2.activation = "sigmoid"
    node2.aggregation = "sum"

    genome.nodes[0] = node0
    genome.nodes[1] = node1
    genome.nodes[2] = node2

    # Connections: input -1 -> hidden 1, hidden 1 -> output 0.
    conn1_key = (-1, 1)
    conn1 = DefaultConnectionGene(conn1_key, innovation=0)
    conn1.weight = 0.5
    conn1.enabled = True

    conn2_key = (1, 0)
    conn2 = DefaultConnectionGene(conn2_key, innovation=1)
    conn2.weight = 1.5
    conn2.enabled = True

    genome.connections[conn1_key] = conn1
    genome.connections[conn2_key] = conn2

    time_constant = 0.01
    net = neat.ctrnn.CTRNN.create(genome, config, time_constant)

    genome_config = config.genome_config

    # Input and output node lists should come from the genome config.
    assert net.input_nodes == genome_config.input_keys
    assert net.output_nodes == genome_config.output_keys

    # Only nodes that are actually required for the outputs should have node_evals.
    assert set(net.node_evals.keys()) == {0, 1}

    ne_hidden = net.node_evals[1]
    assert ne_hidden.time_constant == time_constant
    assert ne_hidden.bias == node1.bias
    assert ne_hidden.response == node1.response
    assert ne_hidden.activation is genome_config.activation_defs.get(node1.activation)
    assert ne_hidden.aggregation is genome_config.aggregation_function_defs.get(node1.aggregation)
    assert ne_hidden.links == [(-1, 0.5)]

    ne_output = net.node_evals[0]
    assert ne_output.time_constant == time_constant
    assert ne_output.bias == node0.bias
    assert ne_output.response == node0.response
    assert ne_output.activation is genome_config.activation_defs.get(node0.activation)
    assert ne_output.aggregation is genome_config.aggregation_function_defs.get(node0.aggregation)
    assert ne_output.links == [(1, 1.5)]


if __name__ == "__main__":
    # Allow running this module directly for quick manual checks.
    test_basic_two_neuron_dynamics()
    test_reset_and_deterministic_trajectory()
    test_advance_input_validation()
    test_ctrnn_create_from_genome_prunes_and_builds_expected_structure()
