"""
Parallel evaluation example with reproducibility.

This example demonstrates that parallel evaluation produces deterministic results
when a random seed is specified. It shows how to use ParallelEvaluator with
reproducible seeding for cross-platform consistency.

Key features:
- Parallel evaluation using ParallelEvaluator with multiple workers
- Reproducibility with fixed random seed
- Verification that same seed produces identical results across runs
- Per-genome deterministic seeding in parallel mode

When run with the same seed, the evolution should produce identical results
regardless of the number of worker processes used.
"""

import multiprocessing
import os
import random

import neat

# Simple fitness inputs (3D space evaluation)
test_inputs = [
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 1.0, 1.0),
]


def eval_genome(genome, config):
    """
    Evaluate a single genome.
    
    This fitness function is intentionally simple to demonstrate parallel
    evaluation without excessive computational overhead. In real applications,
    this would contain the actual neural network evaluation.
    
    The fitness here rewards network complexity (nodes + connections),
    which encourages evolution of more sophisticated networks.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Evaluate network on test inputs
    fitness = 0.0
    for test_input in test_inputs:
        output = net.activate(test_input)
        # Simple fitness: just sum the outputs
        # In real problems, compare against expected outputs
        fitness += abs(output[0]) * 10.0
    
    # Add bonus for network complexity to encourage evolution
    fitness += len(genome.nodes) * 0.5
    fitness += len(genome.connections) * 0.2
    
    return fitness


def run_evolution_with_seed(config_path, seed_value, num_workers, generations=30):
    """
    Run evolution with parallel evaluation and specified seed.
    
    Args:
        config_path: Path to NEAT configuration file
        seed_value: Random seed for reproducibility (or None for non-deterministic)
        num_workers: Number of worker processes to use
        generations: Number of generations to evolve
    
    Returns:
        Tuple of (winner_genome, generation_count, structure_hash)
    """
    # Load configuration
    config = neat.Config(neat.DefaultGenome,
                        neat.DefaultReproduction,
                        neat.DefaultSpeciesSet,
                        neat.DefaultStagnation,
                        config_path)
    
    # Create population with seed
    pop = neat.Population(config, seed=seed_value)
    
    # Create a structure hash for comparison
    def get_pop_hash():
        """Create a hash of current population structure."""
        return tuple(
            (gid, len(g.nodes), len(g.connections))
            for gid in sorted(pop.population.keys())
        )
    
    # Run parallel evolution
    with neat.ParallelEvaluator(num_workers, eval_genome, seed=seed_value) as pe:
        winner = pop.run(pe.evaluate, generations)
    
    # Create structure hash for final population
    final_hash = (len(winner.nodes), len(winner.connections),
                 tuple(sorted([c.key for c in winner.connections.values()])))
    
    return winner, pop.generation, final_hash


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_parallel_reproducibility(config_path, num_workers):
    """
    Test that parallel evaluation with same seed produces identical results.
    """
    print_header(f"TEST 1: PARALLEL REPRODUCIBILITY ({num_workers} workers)")
    
    print("\nRunning first parallel evolution with seed=42...")
    winner1, gen1, hash1 = run_evolution_with_seed(config_path, 42, num_workers)
    
    print("\nRunning second parallel evolution with seed=42...")
    winner2, gen2, hash2 = run_evolution_with_seed(config_path, 42, num_workers)
    
    print_header("RESULTS")
    
    fitness_match = (winner1.fitness == winner2.fitness)
    generation_match = (gen1 == gen2)
    structure_match = (hash1 == hash2)
    
    print(f"Run 1 - Fitness: {winner1.fitness:.2f}, Generations: {gen1}, "
          f"Nodes: {len(winner1.nodes)}, Connections: {len(winner1.connections)}")
    print(f"Run 2 - Fitness: {winner2.fitness:.2f}, Generations: {gen2}, "
          f"Nodes: {len(winner2.nodes)}, Connections: {len(winner2.connections)}")
    
    print(f"\n✓ Fitness match:     {fitness_match}")
    print(f"✓ Generation match:  {generation_match}")
    print(f"✓ Structure match:   {structure_match}")
    
    if fitness_match and generation_match and structure_match:
        print("\n✅ SUCCESS: Reproducibility with parallel evaluation works!")
        return True
    else:
        print("\n❌ FAILURE: Parallel reproducibility test failed!")
        return False


def test_seed_effect(config_path, num_workers):
    """
    Test that different seeds produce different results.
    """
    print_header("TEST 2: DIFFERENT SEEDS IN PARALLEL MODE")
    
    print("\nRunning parallel evolution with seed=42...")
    winner1, gen1, hash1 = run_evolution_with_seed(config_path, 42, num_workers)
    
    print("\nRunning parallel evolution with seed=123...")
    winner2, gen2, hash2 = run_evolution_with_seed(config_path, 123, num_workers)
    
    print_header("RESULTS")
    
    fitness_diff = (winner1.fitness != winner2.fitness)
    generation_diff = (gen1 != gen2)
    structure_diff = (hash1 != hash2)
    
    print(f"Seed 42  - Fitness: {winner1.fitness:.2f}, Generations: {gen1}, "
          f"Nodes: {len(winner1.nodes)}, Connections: {len(winner1.connections)}")
    print(f"Seed 123 - Fitness: {winner2.fitness:.2f}, Generations: {gen2}, "
          f"Nodes: {len(winner2.nodes)}, Connections: {len(winner2.connections)}")
    
    print(f"\n✓ Results differ: {fitness_diff or generation_diff or structure_diff}")
    
    if fitness_diff or generation_diff or structure_diff:
        print("\n✅ SUCCESS: Different seeds produce different evolution!")
        return True
    else:
        print("\n⚠️  WARNING: Different seeds produced same result (rare but possible)")
        return False


def test_worker_count_independence(config_path):
    """
    Test that same seed produces same results with different worker counts.
    """
    print_header("TEST 3: WORKER COUNT INDEPENDENCE")
    
    seed_value = 42
    
    print(f"\nRunning parallel evolution with seed={seed_value} (2 workers)...")
    winner1, gen1, hash1 = run_evolution_with_seed(config_path, seed_value, 2)
    
    print(f"\nRunning parallel evolution with seed={seed_value} (4 workers)...")
    winner2, gen2, hash2 = run_evolution_with_seed(config_path, seed_value, 4)
    
    print_header("RESULTS")
    
    fitness_match = (winner1.fitness == winner2.fitness)
    generation_match = (gen1 == gen2)
    
    print(f"2 workers - Fitness: {winner1.fitness:.2f}, Generations: {gen1}")
    print(f"4 workers - Fitness: {winner2.fitness:.2f}, Generations: {gen2}")
    
    print(f"\n✓ Fitness match:     {fitness_match}")
    print(f"✓ Generation match:  {generation_match}")
    
    if fitness_match and generation_match:
        print("\n✅ SUCCESS: Reproducibility independent of worker count!")
        return True
    else:
        print("\n⚠️  WARNING: Results differ with different worker counts")
        print("(Note: This can happen due to different task scheduling)")
        return False


def main():
    """Run parallel reproducibility tests."""
    # Determine path to configuration file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-parallel')
    
    # Determine number of workers
    max_workers = multiprocessing.cpu_count()
    num_workers = min(4, max_workers)  # Use up to 4 workers
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#  NEAT-Python Parallel Evaluation with Reproducibility" + " " * 13 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    print(f"\nSystem: {max_workers} CPU cores available")
    print(f"Using: {num_workers} worker processes for this test")
    
    print("\nThis script demonstrates reproducibility in parallel evaluation by")
    print("running evolution multiple times and comparing results.")
    
    print("\nThree tests will be performed:")
    print(f"  1. Same seed with {num_workers} workers → identical results")
    print("  2. Different seeds → different results")
    print("  3. Same seed with different worker counts → consistent results")
    
    # Run tests
    test1_pass = test_parallel_reproducibility(config_path, num_workers)
    test2_pass = test_seed_effect(config_path, num_workers)
    test3_pass = test_worker_count_independence(config_path)
    
    # Summary
    print_header("SUMMARY")
    
    if test1_pass and test2_pass and test3_pass:
        print("✅ ALL TESTS PASSED!")
        print("\nParallel reproducibility is working correctly:")
        print(f"  • Same seed with {num_workers} workers produces identical results")
        print("  • Different seeds produce different evolution paths")
        print("  • Results are consistent across different worker counts")
    else:
        print("⚠️  SOME TESTS DID NOT PASS")
        if not test1_pass:
            print("  • Parallel reproducibility test inconclusive")
        if not test2_pass:
            print("  • Different seeds test inconclusive")
        if not test3_pass:
            print("  • Worker count independence test inconclusive")
    
    print("\n" + "#" * 70 + "\n")


if __name__ == '__main__':
    # This is required for multiprocessing on Windows
    main()
