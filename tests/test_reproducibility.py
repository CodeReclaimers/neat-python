"""
Tests for reproducibility functionality.

Tests verify that:
- Same seed produces identical results
- Different seeds produce different results
- Backward compatibility (no seed works as before)
- Config file seed works
- Population parameter seed overrides config seed
- Parallel evaluation is reproducible
"""

import unittest
import os
import random
import tempfile
import shutil
import neat


class TestBasicReproducibility(unittest.TestCase):
    """Tests for basic (serial) reproducibility."""
    
    def setUp(self):
        """Set up test configuration."""
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
    
    def eval_genomes_simple(self, genomes, config):
        """Simple fitness function for testing."""
        for genome_id, genome in genomes:
            genome.fitness = 1.0
    
    def eval_genomes_deterministic(self, genomes, config):
        """Fitness function that doesn't use random but varies by genome structure."""
        for genome_id, genome in genomes:
            # Deterministic fitness based on genome structure
            genome.fitness = float(len(genome.connections)) * 0.1 + float(len(genome.nodes)) * 0.05
    
    def test_same_seed_produces_same_results(self):
        """
        Test that same seed produces identical evolution.
        
        This is the core reproducibility test: running evolution twice
        with the same seed should produce exactly the same results.
        """
        # Run 1
        random.seed(42)  # Reset global random state
        pop1 = neat.Population(self.config, seed=42)
        pop1.run(self.eval_genomes_simple, 10)
        
        # Get final population state
        final_keys_1 = sorted(pop1.population.keys())
        final_fitness_1 = [pop1.population[k].fitness for k in final_keys_1]
        final_connections_1 = [len(pop1.population[k].connections) for k in final_keys_1]
        
        # Run 2
        random.seed(42)  # Reset global random state
        pop2 = neat.Population(self.config, seed=42)
        pop2.run(self.eval_genomes_simple, 10)
        
        # Get final population state
        final_keys_2 = sorted(pop2.population.keys())
        final_fitness_2 = [pop2.population[k].fitness for k in final_keys_2]
        final_connections_2 = [len(pop2.population[k].connections) for k in final_keys_2]
        
        # Verify identical results
        self.assertEqual(final_keys_1, final_keys_2,
                        "Genome IDs should be identical")
        self.assertEqual(final_fitness_1, final_fitness_2,
                        "Fitness values should be identical")
        self.assertEqual(final_connections_1, final_connections_2,
                        "Connection counts should be identical")
    
    def test_different_seeds_produce_different_results(self):
        """
        Test that different seeds produce different evolution.
        
        This verifies that the seed is actually being used and affecting
        the evolution process.
        """
        # Use fitness function with variation to ensure evolution happens
        def eval_with_variation(genomes, config):
            for genome_id, genome in genomes:
                # Fitness varies based on structure and uses random
                import random
                genome.fitness = random.random() + len(genome.connections) * 0.1
        
        # Run with seed 42
        pop1 = neat.Population(self.config, seed=42)
        pop1.run(eval_with_variation, 20)
        
        # Collect fitness values (more reliable indicator than structure)
        fitness_sum_1 = sum(g.fitness for g in pop1.population.values())
        
        # Run with seed 123
        pop2 = neat.Population(self.config, seed=123)
        pop2.run(eval_with_variation, 20)
        
        fitness_sum_2 = sum(g.fitness for g in pop2.population.values())
        
        # Fitness sums should be different (with high probability)
        self.assertNotAlmostEqual(fitness_sum_1, fitness_sum_2, places=5,
                                 msg="Different seeds should produce different results")
    
    def test_no_seed_still_works(self):
        """
        Test backward compatibility - no seed works as before.
        
        Existing code without seed parameter should continue to work.
        """
        # No seed parameter - should work without errors
        pop = neat.Population(self.config)
        winner = pop.run(self.eval_genomes_simple, 5)
        
        # Verify population evolved successfully
        self.assertGreater(len(pop.population), 0,
                          "Population should have genomes")
        self.assertIsNotNone(winner, "Should return a winner genome")
    
    def test_seed_from_config(self):
        """
        Test that seed can be loaded from config file.
        
        Verifies the config file parsing works correctly for seed parameter.
        """
        # Create a temporary config file with seed
        temp_dir = tempfile.mkdtemp(prefix='neat_reproducibility_test_')
        try:
            # Copy test configuration
            local_dir = os.path.dirname(__file__)
            source_config = os.path.join(local_dir, 'test_configuration')
            temp_config = os.path.join(temp_dir, 'config_with_seed')
            
            # Add seed to config
            with open(source_config, 'r') as f_in:
                config_content = f_in.read()
            
            # Insert seed parameter in NEAT section
            config_content = config_content.replace(
                'no_fitness_termination         = False',
                'no_fitness_termination         = False\nseed                         = 42'
            )
            
            with open(temp_config, 'w') as f_out:
                f_out.write(config_content)
            
            # Load config with seed
            config_with_seed = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                temp_config
            )
            
            # Verify seed was loaded
            self.assertEqual(config_with_seed.seed, 42,
                           "Seed should be loaded from config file")
            
            # Verify it works in Population
            random.seed(99)  # Different initial state
            pop1 = neat.Population(config_with_seed)
            pop1.run(self.eval_genomes_simple, 5)
            
            random.seed(99)  # Same initial state
            pop2 = neat.Population(config_with_seed)
            pop2.run(self.eval_genomes_simple, 5)
            
            # Should produce same results
            final_keys_1 = sorted(pop1.population.keys())
            final_keys_2 = sorted(pop2.population.keys())
            self.assertEqual(final_keys_1, final_keys_2,
                           "Config seed should produce reproducible results")
        
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def test_seed_parameter_overrides_config(self):
        """
        Test that Population seed parameter overrides config seed.
        
        This ensures the precedence order is correct: parameter > config.
        """
        # Create a temporary config file with seed=100
        temp_dir = tempfile.mkdtemp(prefix='neat_reproducibility_test_')
        try:
            local_dir = os.path.dirname(__file__)
            source_config = os.path.join(local_dir, 'test_configuration')
            temp_config = os.path.join(temp_dir, 'config_with_seed')
            
            with open(source_config, 'r') as f_in:
                config_content = f_in.read()
            
            config_content = config_content.replace(
                'no_fitness_termination         = False',
                'no_fitness_termination         = False\nseed                         = 100'
            )
            
            with open(temp_config, 'w') as f_out:
                f_out.write(config_content)
            
            config_with_seed = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                temp_config
            )
            
            # Use fitness function with variation to see differences
            def eval_with_variation(genomes, config):
                for genome_id, genome in genomes:
                    import random
                    genome.fitness = random.random() + len(genome.connections) * 0.1
            
            # Run with config seed (100)
            random.seed(42)
            pop1 = neat.Population(config_with_seed)
            pop1.run(eval_with_variation, 10)
            fitness_1 = [g.fitness for g in pop1.population.values()]
            
            # Run with parameter seed (200) - should override config seed (100)
            random.seed(42)
            pop2 = neat.Population(config_with_seed, seed=200)
            pop2.run(eval_with_variation, 10)
            fitness_2 = [g.fitness for g in pop2.population.values()]
            
            # Results should be different (parameter seed overrode config seed)
            # Compare fitness distributions
            self.assertNotAlmostEqual(sum(fitness_1), sum(fitness_2), places=5,
                                    msg="Parameter seed should override config seed")
            
            # Verify parameter seed produces reproducible results
            random.seed(42)
            pop3 = neat.Population(config_with_seed, seed=200)
            pop3.run(eval_with_variation, 10)
            fitness_3 = [g.fitness for g in pop3.population.values()]
            
            self.assertAlmostEqual(sum(fitness_2), sum(fitness_3), places=10,
                                 msg="Parameter seed should produce reproducible results")
        
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    


# Module-level functions for parallel testing (must be picklable)
def eval_genome_simple_parallel(genome, config):
    """Simple fitness function for parallel testing."""
    return 1.0


def eval_genome_with_random_parallel(genome, config):
    """Fitness function that uses random numbers."""
    # Use random in fitness evaluation
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = random.random() + len(genome.connections) * 0.1
    return fitness


class TestParallelReproducibility(unittest.TestCase):
    """Tests for parallel evaluation reproducibility."""
    
    def setUp(self):
        """Set up test configuration."""
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
    
    def test_parallel_same_seed_produces_same_results(self):
        """
        Test that parallel evaluation is reproducible with same seed.
        
        This verifies the per-genome seeding strategy works correctly.
        """
        # Run 1
        random.seed(42)
        with neat.ParallelEvaluator(2, eval_genome_simple_parallel, seed=42) as pe:
            pop1 = neat.Population(self.config, seed=42)
            pop1.run(pe.evaluate, 10)
        
        final_keys_1 = sorted(pop1.population.keys())
        final_connections_1 = [len(pop1.population[k].connections) for k in final_keys_1]
        
        # Run 2
        random.seed(42)
        with neat.ParallelEvaluator(2, eval_genome_simple_parallel, seed=42) as pe:
            pop2 = neat.Population(self.config, seed=42)
            pop2.run(pe.evaluate, 10)
        
        final_keys_2 = sorted(pop2.population.keys())
        final_connections_2 = [len(pop2.population[k].connections) for k in final_keys_2]
        
        # Should be identical
        self.assertEqual(final_keys_1, final_keys_2,
                        "Parallel execution should produce identical genome IDs")
        self.assertEqual(final_connections_1, final_connections_2,
                        "Parallel execution should produce identical structures")
    
    def test_parallel_different_seeds(self):
        """
        Test that different seeds produce different results in parallel.
        
        Verifies the seed is actually being used in parallel mode.
        """
        # Run with seed 42 - use fitness function with randomness
        with neat.ParallelEvaluator(2, eval_genome_with_random_parallel, seed=42) as pe:
            pop1 = neat.Population(self.config, seed=42)
            pop1.run(pe.evaluate, 15)
        
        fitness_sum_1 = sum(g.fitness for g in pop1.population.values())
        
        # Run with seed 123
        with neat.ParallelEvaluator(2, eval_genome_with_random_parallel, seed=123) as pe:
            pop2 = neat.Population(self.config, seed=123)
            pop2.run(pe.evaluate, 15)
        
        fitness_sum_2 = sum(g.fitness for g in pop2.population.values())
        
        # Results should differ
        self.assertNotAlmostEqual(fitness_sum_1, fitness_sum_2, places=5,
                                 msg="Different seeds should produce different results in parallel")
    
    def test_parallel_no_seed_still_works(self):
        """
        Test backward compatibility in parallel mode.
        
        Parallel evaluation without seed should work as before.
        """
        # No seed - should work
        with neat.ParallelEvaluator(2, eval_genome_simple_parallel) as pe:
            pop = neat.Population(self.config)
            pop.run(pe.evaluate, 5)
        
        self.assertGreater(len(pop.population), 0,
                          "Parallel execution should work without seed")
    
    def test_parallel_with_random_fitness(self):
        """
        Test that fitness functions using random are reproducible.
        
        This tests the per-genome seeding where each genome gets a
        deterministic but unique seed.
        """
        # Run 1
        random.seed(99)
        with neat.ParallelEvaluator(2, eval_genome_with_random_parallel, seed=42) as pe:
            pop1 = neat.Population(self.config, seed=42)
            pop1.run(pe.evaluate, 10)
        
        fitness_1 = {gid: genome.fitness for gid, genome in pop1.population.items()}
        
        # Run 2
        random.seed(99)
        with neat.ParallelEvaluator(2, eval_genome_with_random_parallel, seed=42) as pe:
            pop2 = neat.Population(self.config, seed=42)
            pop2.run(pe.evaluate, 10)
        
        fitness_2 = {gid: genome.fitness for gid, genome in pop2.population.items()}
        
        # Fitness values should be identical
        self.assertEqual(set(fitness_1.keys()), set(fitness_2.keys()),
                        "Should have same genome IDs")
        
        for gid in fitness_1:
            self.assertAlmostEqual(fitness_1[gid], fitness_2[gid], places=10,
                                  msg=f"Genome {gid} fitness should be identical")


if __name__ == '__main__':
    unittest.main()
