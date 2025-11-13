"""
Reproducible XOR example demonstrating seed-based deterministic evolution.

This example shows how to use seeds for reproducible evolution with NEAT, 
and verifies that the same seed produces identical results across multiple runs.

Key features:
- Same seed produces identical evolution (reproducible)
- Different seeds produce different evolution (non-deterministic variation)
- Backward compatible (no seed works as before)
- Works with both serial and parallel evaluation
"""

import os
import random
import neat

# 2-input XOR inputs and expected outputs
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genomes(genomes, config):
    """Evaluate fitness of genomes on XOR problem."""
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


def run_with_seed(config_path, seed, generations=50):
    """
    Run evolution with specified seed.
    
    Returns a tuple of (winner_genome, generation_count, genome_structure_hash)
    """
    # Create config
    config = neat.Config(neat.DefaultGenome,
                        neat.DefaultReproduction,
                        neat.DefaultSpeciesSet,
                        neat.DefaultStagnation,
                        config_path)
    
    # Create population with seed - this enables reproducibility
    pop = neat.Population(config, seed=seed)
    
    # Don't show output for cleaner comparison
    # (reduce noise in reproducibility verification)
    
    # Run evolution
    winner = pop.run(eval_genomes, generations)
    
    # Create a simple hash of genome structure for comparison
    # (fitness alone isn't enough - structure should be identical)
    structure_hash = (len(winner.nodes), len(winner.connections),
                     tuple(sorted([c.key for c in winner.connections.values()])))
    
    return winner, pop.generation, structure_hash


def print_separator(title):
    """Print a formatted section separator."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_reproducibility(config_path):
    """
    Test 1: Verify that the same seed produces identical evolution.
    
    This is the core reproducibility guarantee: running evolution twice
    with the same seed should produce exactly the same results.
    """
    print_separator("TEST 1: REPRODUCIBILITY VERIFICATION")
    print("Running evolution with seed=42 (first run)...")
    print("This may take a moment...")
    winner1, gen1, hash1 = run_with_seed(config_path, seed=42)
    
    print("\nRunning evolution with seed=42 (second run)...")
    print("This may take a moment...")
    winner2, gen2, hash2 = run_with_seed(config_path, seed=42)
    
    # Compare results
    print_separator("RESULTS")
    
    fitness_match = (winner1.fitness == winner2.fitness)
    generation_match = (gen1 == gen2)
    structure_match = (hash1 == hash2)
    
    print(f"Run 1 - Fitness: {winner1.fitness:.6f}, Generations: {gen1}, "
          f"Nodes: {len(winner1.nodes)}, Connections: {len(winner1.connections)}")
    print(f"Run 2 - Fitness: {winner2.fitness:.6f}, Generations: {gen2}, "
          f"Nodes: {len(winner2.nodes)}, Connections: {len(winner2.connections)}")
    
    print(f"\n✓ Fitness match:     {fitness_match}")
    print(f"✓ Generation match:  {generation_match}")
    print(f"✓ Structure match:   {structure_match}")
    
    if fitness_match and generation_match and structure_match:
        print("\n✅ SUCCESS: Reproducibility test PASSED!")
        print("   Same seed produces identical evolution results.")
        return True
    else:
        print("\n❌ FAILURE: Reproducibility test FAILED!")
        print("   Identical seeds produced different results.")
        return False


def test_different_seeds(config_path):
    """
    Test 2: Verify that different seeds produce different evolution.
    
    This verifies that the seed is actually affecting the evolution process
    and that different seeds lead to different evolutionary paths.
    """
    print_separator("TEST 2: DIFFERENT SEEDS COMPARISON")
    
    print("Running evolution with seed=42...")
    winner1, gen1, hash1 = run_with_seed(config_path, seed=42)
    
    print("Running evolution with seed=123...")
    winner2, gen2, hash2 = run_with_seed(config_path, seed=123)
    
    # Compare results
    print_separator("RESULTS")
    
    fitness_diff = (winner1.fitness != winner2.fitness)
    generation_diff = (gen1 != gen2)
    structure_diff = (hash1 != hash2)
    
    print(f"Seed 42  - Fitness: {winner1.fitness:.6f}, Generations: {gen1}, "
          f"Nodes: {len(winner1.nodes)}, Connections: {len(winner1.connections)}")
    print(f"Seed 123 - Fitness: {winner2.fitness:.6f}, Generations: {gen2}, "
          f"Nodes: {len(winner2.nodes)}, Connections: {len(winner2.connections)}")
    
    print(f"\n✓ Fitness different:     {fitness_diff}")
    print(f"✓ Generation different:  {generation_diff}")
    print(f"✓ Structure different:   {structure_diff}")
    
    # At least some aspects should be different
    if fitness_diff or generation_diff or structure_diff:
        print("\n✅ SUCCESS: Different seeds test PASSED!")
        print("   Different seeds produced different evolution paths.")
        return True
    else:
        print("\n⚠️  WARNING: Different seeds test inconclusive")
        print("   This is rare but possible if both seeds led to same solution.")
        return False


def demonstrate_backward_compatibility(config_path):
    """
    Test 3: Verify backward compatibility - no seed works as before.
    
    Existing code without seed parameter should continue to work without errors.
    """
    print_separator("TEST 3: BACKWARD COMPATIBILITY")
    
    print("Running evolution WITHOUT seed parameter...")
    print("(This should work like the original example)")
    
    try:
        config = neat.Config(neat.DefaultGenome,
                           neat.DefaultReproduction,
                           neat.DefaultSpeciesSet,
                           neat.DefaultStagnation,
                           config_path)
        
        # No seed parameter - should work fine
        pop = neat.Population(config)
        winner = pop.run(eval_genomes, 10)
        
        print_separator("RESULTS")
        print(f"Evolution completed successfully after {pop.generation} generations")
        print(f"Winner fitness: {winner.fitness:.6f}")
        print(f"Winner structure: {len(winner.nodes)} nodes, {len(winner.connections)} connections")
        print("\n✅ SUCCESS: Backward compatibility test PASSED!")
        print("   Code without seed parameter works as before.")
        return True
    except Exception as e:
        print_separator("ERROR")
        print(f"❌ FAILURE: {e}")
        return False


def main():
    """Run all reproducibility tests."""
    # Determine path to configuration file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#  NEAT-Python Reproducibility Verification" + " " * 25 + "#")
    print("#  XOR Example with Seed-based Deterministic Evolution" + " " * 14 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    print("\nThis script demonstrates reproducibility in NEAT-Python by running")
    print("the same evolution multiple times with different seeds and comparing")
    print("the results.")
    
    print("\nThree tests will be performed:")
    print("  1. Same seed → identical results (reproducibility)")
    print("  2. Different seeds → different results (variation)")
    print("  3. No seed → works like before (backward compatibility)")
    
    # Run tests
    test1_pass = test_reproducibility(config_path)
    test2_pass = test_different_seeds(config_path)
    test3_pass = demonstrate_backward_compatibility(config_path)
    
    # Summary
    print_separator("SUMMARY")
    
    if test1_pass and test2_pass and test3_pass:
        print("✅ ALL TESTS PASSED!")
        print("\nReproducibility is working correctly:")
        print("  • Same seed produces identical evolution")
        print("  • Different seeds produce different evolution")
        print("  • Backward compatibility maintained")
    else:
        print("⚠️  SOME TESTS DID NOT PASS")
        if not test1_pass:
            print("  • Reproducibility test failed")
        if not test2_pass:
            print("  • Different seeds test inconclusive")
        if not test3_pass:
            print("  • Backward compatibility test failed")
    
    print("\n" + "#" * 70 + "\n")


if __name__ == '__main__':
    main()
