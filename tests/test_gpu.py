"""
Tests for the GPU-accelerated CTRNN and Izhikevich evaluators.

Packing tests (genome-to-tensor conversion) run on CPU only and require NumPy.
Numerical equivalence tests require CuPy and a GPU.
"""

import math
import os

import pytest

import neat
from neat.genes import DefaultNodeGene, DefaultConnectionGene

# Optional imports — skip tests if not available.
np = pytest.importorskip("numpy")

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

requires_gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")

LOCAL_DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctrnn_config():
    config_path = os.path.join(LOCAL_DIR, "test_configuration_gpu_ctrnn")
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


def _make_iznn_config():
    config_path = os.path.join(LOCAL_DIR, "test_configuration_iznn")
    return neat.Config(
        neat.iznn.IZGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


def _make_simple_ctrnn_genome(config, genome_id=1, bias=0.5, response=1.0,
                               tau=1.0, w_in1=1.0, w_in2=0.5,
                               activation='tanh', add_hidden=False):
    """
    Create a simple CTRNN genome with 2 inputs, 1 output, optionally 1 hidden.
    All connections enabled, sum aggregation.
    """
    gc = config.genome_config
    genome = neat.DefaultGenome(genome_id)

    # Output node.
    node0 = DefaultNodeGene(0)
    node0.bias = bias
    node0.response = response
    node0.activation = activation
    node0.aggregation = 'sum'
    node0.time_constant = tau
    genome.nodes[0] = node0

    if add_hidden:
        node1 = DefaultNodeGene(1)
        node1.bias = 0.1
        node1.response = 1.0
        node1.activation = activation
        node1.aggregation = 'sum'
        node1.time_constant = tau * 0.5
        genome.nodes[1] = node1

        # input -1 -> hidden 1
        c1 = DefaultConnectionGene((-1, 1), innovation=0)
        c1.weight = w_in1
        c1.enabled = True
        genome.connections[c1.key] = c1

        # input -2 -> hidden 1
        c2 = DefaultConnectionGene((-2, 1), innovation=1)
        c2.weight = w_in2
        c2.enabled = True
        genome.connections[c2.key] = c2

        # hidden 1 -> output 0
        c3 = DefaultConnectionGene((1, 0), innovation=2)
        c3.weight = 1.5
        c3.enabled = True
        genome.connections[c3.key] = c3
    else:
        # input -1 -> output 0
        c1 = DefaultConnectionGene((-1, 0), innovation=0)
        c1.weight = w_in1
        c1.enabled = True
        genome.connections[c1.key] = c1

        # input -2 -> output 0
        c2 = DefaultConnectionGene((-2, 0), innovation=1)
        c2.weight = w_in2
        c2.enabled = True
        genome.connections[c2.key] = c2

    return genome


def _make_simple_iznn_genome(config, genome_id=1, bias=0.0,
                              a=0.02, b=0.2, c=-65.0, d=8.0,
                              w_in1=10.0, w_in2=5.0):
    """Create a simple Izhikevich genome: 2 inputs, 2 outputs."""
    gc = config.genome_config
    genome = neat.iznn.IZGenome(genome_id)

    # Output nodes.
    for out_key in gc.output_keys:
        node = neat.iznn.IZNodeGene(out_key)
        node.bias = bias
        node.a = a
        node.b = b
        node.c = c
        node.d = d
        genome.nodes[out_key] = node

    # Connections: both inputs to both outputs.
    innov = 0
    for in_key in gc.input_keys:
        for out_key in gc.output_keys:
            w = w_in1 if in_key == gc.input_keys[0] else w_in2
            conn = DefaultConnectionGene((in_key, out_key), innovation=innov)
            conn.weight = w
            conn.enabled = True
            genome.connections[conn.key] = conn
            innov += 1

    return genome


# ---------------------------------------------------------------------------
# Packing Tests (CPU only, require NumPy)
# ---------------------------------------------------------------------------

class TestCTRNNPacking:
    """Test genome-to-tensor conversion for CTRNN."""

    def test_basic_packing_shapes(self):
        """Packed arrays should have correct shapes."""
        from neat.gpu._padding import pack_ctrnn_population

        config = _make_ctrnn_config()
        g1 = _make_simple_ctrnn_genome(config, genome_id=1)
        g2 = _make_simple_ctrnn_genome(config, genome_id=2, bias=-0.3, w_in1=2.0)
        genomes = [(1, g1), (2, g2)]

        packed = pack_ctrnn_population(genomes, config)

        N = 2
        num_inputs = 2
        num_outputs = 1
        M = packed['max_nodes']
        assert M >= num_inputs + num_outputs

        assert packed['W'].shape == (N, M, M)
        assert packed['bias'].shape == (N, M)
        assert packed['response'].shape == (N, M)
        assert packed['tau'].shape == (N, M)
        assert packed['activation_id'].shape == (N, M)
        assert packed['node_mask'].shape == (N, M)
        assert packed['num_inputs'] == num_inputs
        assert packed['num_outputs'] == num_outputs

        print(f"  max_nodes={M}, W shape={packed['W'].shape}")

    def test_weight_matrix_values(self):
        """Weight matrix entries should match genome connection weights."""
        from neat.gpu._padding import pack_ctrnn_population

        config = _make_ctrnn_config()
        genome = _make_simple_ctrnn_genome(config, w_in1=3.0, w_in2=-1.5)
        genomes = [(1, genome)]

        packed = pack_ctrnn_population(genomes, config)
        W = packed['W'][0]
        key_map = packed['node_key_maps'][0]

        # Connection from input -1 (index 0) to output 0 (index 2)
        src_idx = key_map[-1]
        dst_idx = key_map[0]
        assert abs(W[dst_idx, src_idx] - 3.0) < 1e-7
        print(f"  W[{dst_idx},{src_idx}] = {W[dst_idx, src_idx]} (expected 3.0)")

        src_idx2 = key_map[-2]
        assert abs(W[dst_idx, src_idx2] - (-1.5)) < 1e-7
        print(f"  W[{dst_idx},{src_idx2}] = {W[dst_idx, src_idx2]} (expected -1.5)")

    def test_node_parameters(self):
        """Bias, response, tau should be packed correctly."""
        from neat.gpu._padding import pack_ctrnn_population

        config = _make_ctrnn_config()
        genome = _make_simple_ctrnn_genome(config, bias=1.5, response=2.0, tau=0.5)
        genomes = [(1, genome)]

        packed = pack_ctrnn_population(genomes, config)
        key_map = packed['node_key_maps'][0]
        out_idx = key_map[0]

        assert abs(packed['bias'][0, out_idx] - 1.5) < 1e-7
        assert abs(packed['response'][0, out_idx] - 2.0) < 1e-7
        assert abs(packed['tau'][0, out_idx] - 0.5) < 1e-7
        print(f"  bias={packed['bias'][0, out_idx]}, response={packed['response'][0, out_idx]}, "
              f"tau={packed['tau'][0, out_idx]}")

    def test_hidden_node_packing(self):
        """Genomes with hidden nodes should be packed with correct max_nodes."""
        from neat.gpu._padding import pack_ctrnn_population

        config = _make_ctrnn_config()
        g1 = _make_simple_ctrnn_genome(config, genome_id=1, add_hidden=False)
        g2 = _make_simple_ctrnn_genome(config, genome_id=2, add_hidden=True)
        genomes = [(1, g1), (2, g2)]

        packed = pack_ctrnn_population(genomes, config)

        # g2 has a hidden node, so max_nodes should be num_inputs + num_outputs + 1 = 4
        assert packed['max_nodes'] == 4
        assert packed['node_mask'][1, 3]  # hidden node should be active in g2
        assert not packed['node_mask'][0, 3]  # padding slot in g1 should be inactive
        print(f"  max_nodes={packed['max_nodes']}, g1 mask={packed['node_mask'][0]}, "
              f"g2 mask={packed['node_mask'][1]}")

    def test_disabled_connections_excluded(self):
        """Disabled connections should not appear in weight matrix."""
        from neat.gpu._padding import pack_ctrnn_population

        config = _make_ctrnn_config()
        genome = _make_simple_ctrnn_genome(config, w_in1=5.0, w_in2=3.0)
        # Disable one connection.
        genome.connections[(-2, 0)].enabled = False
        genomes = [(1, genome)]

        packed = pack_ctrnn_population(genomes, config)
        W = packed['W'][0]
        key_map = packed['node_key_maps'][0]

        # Enabled connection should be present.
        assert abs(W[key_map[0], key_map[-1]] - 5.0) < 1e-7
        # Disabled connection should be zero.
        assert abs(W[key_map[0], key_map[-2]]) < 1e-7
        print(f"  Enabled weight: {W[key_map[0], key_map[-1]]}, "
              f"Disabled weight: {W[key_map[0], key_map[-2]]}")

    def test_unsupported_aggregation_raises(self):
        """Non-sum aggregation should raise ValueError."""
        from neat.gpu._padding import pack_ctrnn_population

        config = _make_ctrnn_config()
        genome = _make_simple_ctrnn_genome(config)
        genome.nodes[0].aggregation = 'product'
        genomes = [(1, genome)]

        with pytest.raises(ValueError, match="aggregation.*product.*not supported on GPU"):
            pack_ctrnn_population(genomes, config)

    def test_unsupported_activation_raises(self):
        """Unsupported activation function should raise ValueError."""
        from neat.gpu._padding import pack_ctrnn_population

        config = _make_ctrnn_config()
        genome = _make_simple_ctrnn_genome(config, activation='hat')
        genomes = [(1, genome)]

        with pytest.raises(ValueError, match="activation.*hat.*not supported on GPU"):
            pack_ctrnn_population(genomes, config)

    def test_no_connections_genome(self):
        """A genome with no connections should pack with zero weight matrix."""
        from neat.gpu._padding import pack_ctrnn_population

        config = _make_ctrnn_config()
        genome = neat.DefaultGenome(1)
        node0 = DefaultNodeGene(0)
        node0.bias = 0.0
        node0.response = 1.0
        node0.activation = 'tanh'
        node0.aggregation = 'sum'
        node0.time_constant = 1.0
        genome.nodes[0] = node0
        # No connections at all.
        genomes = [(1, genome)]

        packed = pack_ctrnn_population(genomes, config)
        assert np.all(packed['W'] == 0)
        print(f"  No connections: W all zeros, max_nodes={packed['max_nodes']}")

    def test_input_output_mask_always_true(self):
        """Input and output slots must be active in node_mask for all genomes."""
        from neat.gpu._padding import pack_ctrnn_population

        config = _make_ctrnn_config()
        g1 = _make_simple_ctrnn_genome(config, genome_id=1)
        g2 = _make_simple_ctrnn_genome(config, genome_id=2, add_hidden=True)
        genomes = [(1, g1), (2, g2)]

        packed = pack_ctrnn_population(genomes, config)
        num_inputs = packed['num_inputs']
        num_outputs = packed['num_outputs']

        for g_idx in range(2):
            # Input slots always active.
            assert np.all(packed['node_mask'][g_idx, :num_inputs])
            # Output slots always active.
            assert np.all(packed['node_mask'][g_idx, num_inputs:num_inputs + num_outputs])
        print("  Input/output mask slots are True for all genomes.")


class TestIZNNPacking:
    """Test genome-to-tensor conversion for Izhikevich spiking networks."""

    def test_basic_packing_shapes(self):
        from neat.gpu._padding import pack_iznn_population

        config = _make_iznn_config()
        g1 = _make_simple_iznn_genome(config, genome_id=1)
        g2 = _make_simple_iznn_genome(config, genome_id=2, bias=1.0)
        genomes = [(1, g1), (2, g2)]

        packed = pack_iznn_population(genomes, config)

        N = 2
        M = packed['max_nodes']
        assert packed['W'].shape == (N, M, M)
        assert packed['bias'].shape == (N, M)
        assert packed['a'].shape == (N, M)
        assert packed['b'].shape == (N, M)
        assert packed['c'].shape == (N, M)
        assert packed['d'].shape == (N, M)
        assert packed['num_inputs'] == 2
        assert packed['num_outputs'] == 2
        print(f"  max_nodes={M}, shapes correct")

    def test_izhikevich_parameters(self):
        from neat.gpu._padding import pack_iznn_population

        config = _make_iznn_config()
        genome = _make_simple_iznn_genome(config, a=0.1, b=0.25, c=-55.0, d=4.0)
        genomes = [(1, genome)]

        packed = pack_iznn_population(genomes, config)
        key_map = packed['node_key_maps'][0]

        for out_key in config.genome_config.output_keys:
            idx = key_map[out_key]
            assert abs(packed['a'][0, idx] - 0.1) < 1e-7
            assert abs(packed['b'][0, idx] - 0.25) < 1e-7
            assert abs(packed['c'][0, idx] - (-55.0)) < 1e-7
            assert abs(packed['d'][0, idx] - 4.0) < 1e-7
        print(f"  Izhikevich params packed correctly")


# ---------------------------------------------------------------------------
# GPU Numerical Equivalence Tests (require CuPy)
# ---------------------------------------------------------------------------

@requires_gpu
class TestCTRNNGPUEquivalence:
    """
    Verify that GPU CTRNN evaluation produces results equivalent to the CPU
    CTRNN evaluation.

    Since the CPU uses forward Euler and the GPU uses exponential Euler, the
    results will not be bit-identical at a given dt. Instead we verify that
    both methods converge to the same trajectory as dt decreases.
    """

    def test_single_genome_numerical_agreement(self):
        """
        CPU and GPU should produce nearly identical CTRNN trajectories.

        Both now use exponential Euler integration. The only source of
        difference is float32 (GPU) vs float64 (CPU) precision, plus the
        GPU activation kernel operating on float32. We expect agreement
        to within ~1e-3 relative error over a moderate simulation.
        """
        from neat.gpu._padding import pack_ctrnn_population
        from neat.gpu._cupy_backend import evaluate_ctrnn_batch

        config = _make_ctrnn_config()
        genome = _make_simple_ctrnn_genome(config, bias=0.5, response=1.2,
                                           tau=0.5, w_in1=1.0, w_in2=-0.8)
        genomes = [(1, genome)]

        dt = 0.005
        t_max = 0.5
        num_steps = int(t_max / dt)
        input_vals = [0.3, -0.2]

        # CPU evaluation (float64).
        net = neat.ctrnn.CTRNN.create(genome, config)
        cpu_outputs = []
        for step in range(num_steps):
            out = net.advance(input_vals, dt, dt)
            cpu_outputs.append(out[0])

        # GPU evaluation (float32).
        inputs_np = np.tile(np.array(input_vals, dtype=np.float32),
                            (num_steps, 1))
        packed = pack_ctrnn_population(genomes, config)
        trajectory = evaluate_ctrnn_batch(packed, inputs_np, dt)
        gpu_outputs = trajectory[0, :, 0]

        # Compare full trajectory.
        cpu_arr = np.array(cpu_outputs, dtype=np.float64)
        gpu_arr = np.array(gpu_outputs, dtype=np.float64)
        max_abs_error = np.max(np.abs(cpu_arr - gpu_arr))
        print(f"  dt={dt}, num_steps={num_steps}")
        print(f"  CPU final={cpu_arr[-1]:.8f}, GPU final={gpu_arr[-1]:.8f}")
        print(f"  Max absolute error over trajectory: {max_abs_error:.2e}")

        # float32 accumulation over 100 steps: expect errors < 1e-3.
        assert max_abs_error < 1e-3, (
            f"CPU-GPU disagreement too large: {max_abs_error:.2e}")

    def test_multiple_genomes_same_as_individual(self):
        """
        Batched evaluation of N genomes should produce the same results as
        evaluating each genome individually.
        """
        from neat.gpu._padding import pack_ctrnn_population
        from neat.gpu._cupy_backend import evaluate_ctrnn_batch

        config = _make_ctrnn_config()
        genomes = [
            (1, _make_simple_ctrnn_genome(config, genome_id=1, bias=0.3, w_in1=1.0)),
            (2, _make_simple_ctrnn_genome(config, genome_id=2, bias=-0.5, w_in1=2.0)),
            (3, _make_simple_ctrnn_genome(config, genome_id=3, bias=0.0, w_in1=-1.0,
                                           add_hidden=True)),
        ]

        dt = 0.005
        t_max = 0.2
        num_steps = int(t_max / dt)
        input_vals = [0.5, -0.3]
        inputs_np = np.tile(np.array(input_vals, dtype=np.float32),
                            (num_steps, 1))

        # Batch evaluation.
        packed_all = pack_ctrnn_population(genomes, config)
        traj_all = evaluate_ctrnn_batch(packed_all, inputs_np, dt)

        # Individual evaluation.
        for i, (gid, genome) in enumerate(genomes):
            packed_one = pack_ctrnn_population([(gid, genome)], config)
            traj_one = evaluate_ctrnn_batch(packed_one, inputs_np, dt)

            max_diff = np.max(np.abs(traj_all[i] - traj_one[0]))
            print(f"  Genome {gid}: max batch vs individual diff = {max_diff:.2e}")
            assert max_diff < 1e-6, (
                f"Genome {gid}: batched result differs from individual by {max_diff}")

    def test_response_parameter_effect(self):
        """
        Two genomes with different response values but same weights should
        produce different outputs.
        """
        from neat.gpu._padding import pack_ctrnn_population
        from neat.gpu._cupy_backend import evaluate_ctrnn_batch

        config = _make_ctrnn_config()
        # Use small weights and inputs to avoid tanh saturation, which would
        # mask the effect of different response values.
        g1 = _make_simple_ctrnn_genome(config, genome_id=1, response=0.5,
                                        w_in1=0.3, w_in2=0.1, bias=0.0)
        g2 = _make_simple_ctrnn_genome(config, genome_id=2, response=3.0,
                                        w_in1=0.3, w_in2=0.1, bias=0.0)
        genomes = [(1, g1), (2, g2)]

        dt = 0.005
        num_steps = 50
        inputs_np = np.tile(np.array([0.2, 0.1], dtype=np.float32),
                            (num_steps, 1))

        packed = pack_ctrnn_population(genomes, config)
        traj = evaluate_ctrnn_batch(packed, inputs_np, dt)

        # They should differ.
        diff = np.max(np.abs(traj[0] - traj[1]))
        print(f"  Response 1.0 vs 3.0: max output diff = {diff:.6f}")
        assert diff > 0.01, "Different response values should produce different outputs"

    def test_zero_input_decays_to_activation_of_bias(self):
        """
        With zero inputs and a single output node (no connections from inputs
        that matter), the CTRNN should decay toward activation(bias * response).
        """
        from neat.gpu._padding import pack_ctrnn_population
        from neat.gpu._cupy_backend import evaluate_ctrnn_batch

        config = _make_ctrnn_config()
        # Create genome with zero-weight connections (effectively no input).
        genome = _make_simple_ctrnn_genome(config, bias=0.5, response=1.0,
                                           tau=0.1, w_in1=0.0, w_in2=0.0,
                                           activation='tanh')
        genomes = [(1, genome)]

        dt = 0.001
        num_steps = 5000  # enough time for tau=0.1 to reach steady state
        inputs_np = np.zeros((num_steps, 2), dtype=np.float32)

        packed = pack_ctrnn_population(genomes, config)
        traj = evaluate_ctrnn_batch(packed, inputs_np, dt)

        # Steady state: u* = tanh(bias + response * 0) = tanh(0.5)
        # Actually for CTRNN: du/dt = (-u + z) / tau where z = tanh(bias + response * sum_j(W_ij * u_j))
        # At steady state: u* = z = tanh(bias + response * 0) since W connections are zero
        # But wait, the output node has a self-connection? No — only connections in the genome.
        # With w_in1=0 and w_in2=0, the matmul gives 0 for the output node,
        # so z = tanh(0.5 + 1.0 * 0) = tanh(0.5)
        expected_steady = math.tanh(2.5 * 0.5)  # neat-python tanh clips 2.5*z
        final_val = traj[0, -1, 0]
        print(f"  Final value: {final_val:.6f}, expected tanh(1.25)={expected_steady:.6f}")
        assert abs(final_val - expected_steady) < 0.01


@requires_gpu
class TestIZNNGPUEquivalence:
    """Verify GPU Izhikevich evaluation matches CPU."""

    def test_single_genome_spike_count(self):
        """
        GPU and CPU should produce the same number of spikes (within tolerance)
        for the same genome and inputs.
        """
        from neat.gpu._padding import pack_iznn_population
        from neat.gpu._cupy_backend import evaluate_iznn_batch

        config = _make_iznn_config()
        genome = _make_simple_iznn_genome(config, bias=0.0, w_in1=15.0, w_in2=10.0)
        genomes = [(1, genome)]

        dt = 0.05  # match CPU default
        t_max = 50.0  # 50 ms
        num_steps = int(t_max / dt)

        # Constant input.
        input_vals = [1.0, 0.5]

        # CPU evaluation.
        net = neat.iznn.IZNN.create(genome, config)
        net.set_inputs(input_vals)
        cpu_spikes = [0] * len(config.genome_config.output_keys)
        for step in range(num_steps):
            fired = net.advance(dt)
            for i, f in enumerate(fired):
                if f > 0.5:
                    cpu_spikes[i] += 1

        # GPU evaluation.
        inputs_np = np.tile(np.array(input_vals, dtype=np.float32),
                            (num_steps, 1))
        packed = pack_iznn_population(genomes, config)
        traj = evaluate_iznn_batch(packed, inputs_np, dt, num_steps)

        gpu_spikes = [int(np.sum(traj[0, :, i] > 0.5)) for i in range(traj.shape[2])]

        print(f"  CPU spikes: {cpu_spikes}")
        print(f"  GPU spikes: {gpu_spikes}")

        # Spike counts should match exactly (same numerical method, same dt).
        for i, (cs, gs) in enumerate(zip(cpu_spikes, gpu_spikes)):
            assert cs == gs, (
                f"Output {i}: CPU spikes={cs}, GPU spikes={gs}")

    def test_multiple_genomes_independent(self):
        """Batched evaluation should match individual evaluation."""
        from neat.gpu._padding import pack_iznn_population
        from neat.gpu._cupy_backend import evaluate_iznn_batch

        config = _make_iznn_config()
        genomes = [
            (1, _make_simple_iznn_genome(config, genome_id=1, w_in1=15.0)),
            (2, _make_simple_iznn_genome(config, genome_id=2, w_in1=5.0, bias=2.0)),
        ]

        dt = 0.05
        num_steps = 200
        inputs_np = np.tile(np.array([1.0, 0.5], dtype=np.float32),
                            (num_steps, 1))

        packed_all = pack_iznn_population(genomes, config)
        traj_all = evaluate_iznn_batch(packed_all, inputs_np, dt, num_steps)

        for i, (gid, genome) in enumerate(genomes):
            packed_one = pack_iznn_population([(gid, genome)], config)
            traj_one = evaluate_iznn_batch(packed_one, inputs_np, dt, num_steps)

            max_diff = np.max(np.abs(traj_all[i] - traj_one[0]))
            print(f"  Genome {gid}: max batch vs individual diff = {max_diff:.2e}")
            assert max_diff < 1e-6

    def test_no_input_no_spikes(self):
        """With zero external input and zero bias, neurons should not spike."""
        from neat.gpu._padding import pack_iznn_population
        from neat.gpu._cupy_backend import evaluate_iznn_batch

        config = _make_iznn_config()
        genome = _make_simple_iznn_genome(config, bias=0.0, w_in1=0.0, w_in2=0.0)
        genomes = [(1, genome)]

        dt = 0.05
        num_steps = 500
        inputs_np = np.zeros((num_steps, 2), dtype=np.float32)

        packed = pack_iznn_population(genomes, config)
        traj = evaluate_iznn_batch(packed, inputs_np, dt, num_steps)

        total_spikes = np.sum(traj > 0.5)
        print(f"  Zero input, zero bias: total spikes = {total_spikes}")
        assert total_spikes == 0


# ---------------------------------------------------------------------------
# Evaluator Integration Tests (require CuPy)
# ---------------------------------------------------------------------------

@requires_gpu
class TestGPUEvaluatorIntegration:
    """Test the high-level evaluator classes."""

    def test_ctrnn_evaluator_assigns_fitness(self):
        """GPUCTRNNEvaluator.evaluate should set genome.fitness for all genomes."""
        from neat.gpu.evaluator import GPUCTRNNEvaluator

        config = _make_ctrnn_config()
        genomes = [
            (1, _make_simple_ctrnn_genome(config, genome_id=1, bias=0.3)),
            (2, _make_simple_ctrnn_genome(config, genome_id=2, bias=-0.5)),
        ]

        def input_fn(t, dt):
            return [math.sin(2 * math.pi * t), math.cos(2 * math.pi * t)]

        def fitness_fn(trajectory):
            # Fitness = mean of absolute output values.
            return float(np.mean(np.abs(trajectory)))

        evaluator = GPUCTRNNEvaluator(dt=0.01, t_max=0.5,
                                       input_fn=input_fn, fitness_fn=fitness_fn)
        evaluator.evaluate(genomes, config)

        for gid, genome in genomes:
            assert genome.fitness is not None
            assert isinstance(genome.fitness, float)
            assert genome.fitness >= 0.0
            print(f"  Genome {gid}: fitness = {genome.fitness:.6f}")

    def test_iznn_evaluator_assigns_fitness(self):
        """GPUIZNNEvaluator.evaluate should set genome.fitness for all genomes."""
        from neat.gpu.evaluator import GPUIZNNEvaluator

        config = _make_iznn_config()
        genomes = [
            (1, _make_simple_iznn_genome(config, genome_id=1, w_in1=15.0)),
            (2, _make_simple_iznn_genome(config, genome_id=2, w_in1=5.0)),
        ]

        def input_fn(t, dt):
            return [1.0, 0.5]

        def fitness_fn(trajectory):
            return float(np.sum(trajectory))

        evaluator = GPUIZNNEvaluator(dt=0.05, t_max=25.0,
                                      input_fn=input_fn, fitness_fn=fitness_fn)
        evaluator.evaluate(genomes, config)

        for gid, genome in genomes:
            assert genome.fitness is not None
            assert isinstance(genome.fitness, float)
            print(f"  Genome {gid}: fitness = {genome.fitness:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
