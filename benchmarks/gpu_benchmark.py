#!/usr/bin/env python3
"""
Benchmark comparing CPU vs GPU evaluation for CTRNN and Izhikevich networks.

Usage:
    python benchmarks/gpu_benchmark.py

Requires CuPy and NumPy.
"""

import math
import os
import sys
import time

# Add project root to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import neat
from neat.genes import DefaultNodeGene, DefaultConnectionGene

try:
    import cupy as cp
except ImportError:
    print("CuPy not installed. GPU benchmarks will be skipped.")
    print("Install with: pip install 'neat-python[gpu]'")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration and genome helpers
# ---------------------------------------------------------------------------

def make_ctrnn_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                               'test_configuration_gpu_ctrnn')
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


def make_iznn_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'tests',
                               'test_configuration_iznn')
    return neat.Config(
        neat.iznn.IZGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


def make_ctrnn_genome(config, genome_id, num_hidden=0):
    """Create a CTRNN genome with specified number of hidden nodes."""
    gc = config.genome_config
    genome = neat.DefaultGenome(genome_id)

    # Output node.
    node0 = DefaultNodeGene(0)
    node0.bias = np.random.uniform(-1, 1)
    node0.response = np.random.uniform(0.5, 2.0)
    node0.activation = 'tanh'
    node0.aggregation = 'sum'
    node0.time_constant = np.random.uniform(0.01, 2.0)
    genome.nodes[0] = node0

    innov = 0
    hidden_keys = []
    for h in range(num_hidden):
        key = h + 1
        node = DefaultNodeGene(key)
        node.bias = np.random.uniform(-1, 1)
        node.response = np.random.uniform(0.5, 2.0)
        node.activation = 'tanh'
        node.aggregation = 'sum'
        node.time_constant = np.random.uniform(0.01, 2.0)
        genome.nodes[key] = node
        hidden_keys.append(key)

    # Connect inputs to first layer (hidden or output).
    targets = hidden_keys if hidden_keys else [0]
    for in_key in gc.input_keys:
        for t in targets:
            conn = DefaultConnectionGene((in_key, t), innovation=innov)
            conn.weight = np.random.uniform(-2, 2)
            conn.enabled = True
            genome.connections[conn.key] = conn
            innov += 1

    # Connect hidden to output.
    if hidden_keys:
        for h in hidden_keys:
            conn = DefaultConnectionGene((h, 0), innovation=innov)
            conn.weight = np.random.uniform(-2, 2)
            conn.enabled = True
            genome.connections[conn.key] = conn
            innov += 1

    return genome


def make_iznn_genome(config, genome_id, num_hidden=0):
    """Create an Izhikevich genome."""
    gc = config.genome_config
    genome = neat.iznn.IZGenome(genome_id)

    for out_key in gc.output_keys:
        node = neat.iznn.IZNodeGene(out_key)
        node.bias = np.random.uniform(-5, 5)
        node.a = 0.02
        node.b = 0.2
        node.c = -65.0
        node.d = 8.0
        genome.nodes[out_key] = node

    innov = 0
    hidden_keys = []
    for h in range(num_hidden):
        key = max(gc.output_keys) + 1 + h
        node = neat.iznn.IZNodeGene(key)
        node.bias = np.random.uniform(-5, 5)
        node.a = 0.02
        node.b = 0.2
        node.c = -65.0
        node.d = 8.0
        genome.nodes[key] = node
        hidden_keys.append(key)

    targets = hidden_keys if hidden_keys else gc.output_keys
    for in_key in gc.input_keys:
        for t in targets:
            conn = DefaultConnectionGene((in_key, t), innovation=innov)
            conn.weight = np.random.uniform(-10, 10)
            conn.enabled = True
            genome.connections[conn.key] = conn
            innov += 1

    if hidden_keys:
        for h in hidden_keys:
            for out_key in gc.output_keys:
                conn = DefaultConnectionGene((h, out_key), innovation=innov)
                conn.weight = np.random.uniform(-10, 10)
                conn.enabled = True
                genome.connections[conn.key] = conn
                innov += 1

    return genome


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def benchmark_ctrnn(pop_sizes, num_hidden=3):
    """Benchmark CTRNN CPU vs GPU at various population sizes."""
    from neat.gpu._padding import pack_ctrnn_population
    from neat.gpu._cupy_backend import evaluate_ctrnn_batch

    config = make_ctrnn_config()
    dt = 0.01
    t_max = 1.0
    num_steps = int(t_max / dt)
    input_vals = [0.5, -0.3]
    inputs_np = np.tile(np.array(input_vals, dtype=np.float32), (num_steps, 1))

    print(f"\n{'='*70}")
    print(f"CTRNN Benchmark: dt={dt}, t_max={t_max}, num_steps={num_steps}, "
          f"hidden_nodes={num_hidden}")
    print(f"{'='*70}")
    print(f"{'Pop Size':>10} {'Max Nodes':>10} {'CPU (s)':>10} {'GPU (s)':>10} {'Speedup':>10}")
    print(f"{'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10}")

    for pop_size in pop_sizes:
        np.random.seed(42)
        genomes = [(i, make_ctrnn_genome(config, i, num_hidden=num_hidden))
                   for i in range(pop_size)]

        # CPU timing.
        t0 = time.perf_counter()
        for gid, genome in genomes:
            net = neat.ctrnn.CTRNN.create(genome, config)
            for step in range(num_steps):
                net.advance(input_vals, dt, dt)
        cpu_time = time.perf_counter() - t0

        # GPU timing (include packing + transfer + compute).
        # Warmup.
        packed = pack_ctrnn_population(genomes, config)
        _ = evaluate_ctrnn_batch(packed, inputs_np, dt)
        cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()
        packed = pack_ctrnn_population(genomes, config)
        trajectory = evaluate_ctrnn_batch(packed, inputs_np, dt)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.perf_counter() - t0

        max_nodes = packed['max_nodes']
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')

        print(f"{pop_size:>10d} {max_nodes:>10d} {cpu_time:>10.3f} {gpu_time:>10.3f} "
              f"{speedup:>9.1f}x")


def benchmark_iznn(pop_sizes, num_hidden=3):
    """Benchmark Izhikevich CPU vs GPU at various population sizes."""
    from neat.gpu._padding import pack_iznn_population
    from neat.gpu._cupy_backend import evaluate_iznn_batch

    config = make_iznn_config()
    dt = 0.05
    t_max = 50.0  # 50 ms
    num_steps = int(t_max / dt)
    input_vals = [1.0, 0.5]
    inputs_np = np.tile(np.array(input_vals, dtype=np.float32), (num_steps, 1))

    print(f"\n{'='*70}")
    print(f"Izhikevich Benchmark: dt={dt} ms, t_max={t_max} ms, "
          f"num_steps={num_steps}, hidden_nodes={num_hidden}")
    print(f"{'='*70}")
    print(f"{'Pop Size':>10} {'Max Nodes':>10} {'CPU (s)':>10} {'GPU (s)':>10} {'Speedup':>10}")
    print(f"{'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10}")

    for pop_size in pop_sizes:
        np.random.seed(42)
        genomes = [(i, make_iznn_genome(config, i, num_hidden=num_hidden))
                   for i in range(pop_size)]

        # CPU timing.
        t0 = time.perf_counter()
        for gid, genome in genomes:
            net = neat.iznn.IZNN.create(genome, config)
            net.set_inputs(input_vals)
            for step in range(num_steps):
                net.advance(dt)
        cpu_time = time.perf_counter() - t0

        # GPU timing.
        packed = pack_iznn_population(genomes, config)
        _ = evaluate_iznn_batch(packed, inputs_np, dt, num_steps)
        cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()
        packed = pack_iznn_population(genomes, config)
        trajectory = evaluate_iznn_batch(packed, inputs_np, dt, num_steps)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.perf_counter() - t0

        max_nodes = packed['max_nodes']
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')

        print(f"{pop_size:>10d} {max_nodes:>10d} {cpu_time:>10.3f} {gpu_time:>10.3f} "
              f"{speedup:>9.1f}x")


if __name__ == '__main__':
    pop_sizes = [100, 500, 1000]
    benchmark_ctrnn(pop_sizes)
    benchmark_iznn(pop_sizes)
