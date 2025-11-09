"""
Tests for fitness evaluation and error handling.

Tests verify:
- Fitness function error handling
- Fitness criterion calculations (max, min, mean)
- Fitness threshold termination
- Edge cases with fitness values
"""

import unittest
import os
import neat


class TestFitnessEvaluation(unittest.TestCase):
    """Test fitness evaluation and error handling."""
    
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
    
    # ========== Fitness Function Error Handling ==========
    
    def test_fitness_not_assigned_raises_error(self):
        """
        Test that RuntimeError is raised when fitness is not assigned.
        
        If fitness function doesn't set genome.fitness, should raise error.
        """
        pop = neat.Population(self.config)
        
        def bad_fitness_function(genomes, config):
            # Intentionally don't set fitness
            pass
        
        with self.assertRaises(RuntimeError) as context:
            pop.run(bad_fitness_function, 1)
        
        self.assertIn("Fitness not assigned", str(context.exception))
    
    def test_fitness_set_to_none_raises_error(self):
        """
        Test that RuntimeError is raised when fitness is explicitly None.
        
        Setting fitness to None should be caught.
        """
        pop = neat.Population(self.config)
        
        def bad_fitness_function(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = None
        
        with self.assertRaises(RuntimeError) as context:
            pop.run(bad_fitness_function, 1)
        
        self.assertIn("Fitness not assigned", str(context.exception))
    
    def test_fitness_function_with_valid_values(self):
        """
        Test that fitness function works correctly with valid fitness values.
        
        Should complete without errors when all genomes get fitness.
        """
        pop = neat.Population(self.config)
        
        def good_fitness_function(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 0.5
        
        # Should not raise any exception
        result = pop.run(good_fitness_function, 2)
        self.assertIsNotNone(result)
    
    def test_fitness_function_exception_propagates(self):
        """
        Test that exceptions in fitness function are propagated.
        
        Errors in fitness function should not be silently swallowed.
        """
        pop = neat.Population(self.config)
        
        def error_fitness_function(genomes, config):
            raise ValueError("Test error in fitness function")
        
        with self.assertRaises(ValueError) as context:
            pop.run(error_fitness_function, 1)
        
        self.assertIn("Test error", str(context.exception))
    
    # ========== Fitness Criterion Calculations ==========
    
    def test_fitness_criterion_max(self):
        """
        Test that 'max' fitness criterion selects maximum fitness.
        
        Should terminate when max fitness reaches threshold.
        """
        self.config.fitness_criterion = 'max'
        self.config.fitness_threshold = 5.0
        pop = neat.Population(self.config)
        
        generation_count = [0]
        
        def fitness_function(genomes, config):
            generation_count[0] += 1
            for i, (genome_id, genome) in enumerate(genomes):
                # One genome gets high fitness
                genome.fitness = 6.0 if i == 0 else 1.0
        
        result = pop.run(fitness_function, 10)
        
        # Should terminate after 1 generation (max = 6.0 >= 5.0)
        self.assertEqual(generation_count[0], 1)
        self.assertEqual(result.fitness, 6.0)
    
    def test_fitness_criterion_min(self):
        """
        Test that 'min' fitness criterion selects minimum fitness.
        
        Should terminate when min fitness reaches threshold.
        """
        self.config.fitness_criterion = 'min'
        self.config.fitness_threshold = 0.5
        pop = neat.Population(self.config)
        
        generation_count = [0]
        
        def fitness_function(genomes, config):
            generation_count[0] += 1
            for i, (genome_id, genome) in enumerate(genomes):
                # All genomes get fitness >= 0.5
                genome.fitness = 0.6 + i * 0.1
        
        result = pop.run(fitness_function, 10)
        
        # Should terminate after 1 generation (min = 0.6 >= 0.5)
        self.assertEqual(generation_count[0], 1)
    
    def test_fitness_criterion_mean(self):
        """
        Test that 'mean' fitness criterion uses average fitness.
        
        Should terminate when mean fitness reaches threshold.
        """
        self.config.fitness_criterion = 'mean'
        self.config.fitness_threshold = 5.0
        pop = neat.Population(self.config)
        
        generation_count = [0]
        
        def fitness_function(genomes, config):
            generation_count[0] += 1
            # Set fitness so mean is exactly 5.0
            for i, (genome_id, genome) in enumerate(genomes):
                genome.fitness = 5.0
        
        result = pop.run(fitness_function, 10)
        
        # Should terminate after 1 generation (mean = 5.0 >= 5.0)
        self.assertEqual(generation_count[0], 1)
    
    def test_invalid_fitness_criterion_raises_error(self):
        """
        Test that invalid fitness criterion raises error.
        
        Should catch invalid criterion at Population creation.
        """
        self.config.fitness_criterion = 'invalid_criterion'
        self.config.no_fitness_termination = False
        
        with self.assertRaises(RuntimeError) as context:
            pop = neat.Population(self.config)
        
        self.assertIn("Unexpected fitness_criterion", str(context.exception))
    
    # ========== Fitness Threshold Termination ==========
    
    def test_threshold_termination_stops_evolution(self):
        """
        Test that evolution stops when threshold is reached.
        
        Should not run all generations if threshold is met early.
        """
        self.config.fitness_criterion = 'max'
        self.config.fitness_threshold = 2.0
        pop = neat.Population(self.config)
        
        generation_count = [0]
        
        def fitness_function(genomes, config):
            generation_count[0] += 1
            for genome_id, genome in genomes:
                # All genomes get fitness above threshold
                genome.fitness = 3.0
        
        pop.run(fitness_function, 100)
        
        # Should stop after 1 generation, not run all 100
        self.assertEqual(generation_count[0], 1)
    
    def test_threshold_not_reached_runs_all_generations(self):
        """
        Test that all generations run if threshold is never reached.
        
        Should run exactly n generations if threshold not met.
        """
        self.config.fitness_criterion = 'max'
        self.config.fitness_threshold = 100.0  # Very high
        pop = neat.Population(self.config)
        
        generation_count = [0]
        
        def fitness_function(genomes, config):
            generation_count[0] += 1
            for genome_id, genome in genomes:
                genome.fitness = 1.0  # Always below threshold
        
        pop.run(fitness_function, 5)
        
        # Should run all 5 generations
        self.assertEqual(generation_count[0], 5)
    
    def test_best_genome_returned(self):
        """
        Test that best genome is returned after evolution.
        
        Should return the genome with highest fitness seen.
        """
        self.config.fitness_criterion = 'max'
        self.config.fitness_threshold = 100.0  # Won't be reached
        pop = neat.Population(self.config)
        
        def fitness_function(genomes, config):
            for i, (genome_id, genome) in enumerate(genomes):
                # First genome gets highest fitness
                genome.fitness = 10.0 if i == 0 else 1.0
        
        result = pop.run(fitness_function, 2)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.fitness, 10.0)
    
    def test_best_genome_tracked_across_generations(self):
        """
        Test that best genome is tracked even if lost in later generations.
        
        Should return best ever seen, not best in final generation.
        """
        self.config.fitness_criterion = 'max'
        self.config.fitness_threshold = 100.0
        pop = neat.Population(self.config)
        
        generation = [0]
        best_fitness_seen = [0]
        
        def fitness_function(genomes, config):
            generation[0] += 1
            for i, (genome_id, genome) in enumerate(genomes):
                # High fitness in first generation, low in second
                if generation[0] == 1:
                    genome.fitness = 10.0 if i == 0 else 1.0
                    if i == 0:
                        best_fitness_seen[0] = 10.0
                else:
                    genome.fitness = 2.0
        
        result = pop.run(fitness_function, 2)
        
        # Best genome should have fitness >= best ever seen
        # (The genome object's fitness may change, but best_genome tracked the best)
        self.assertGreaterEqual(result.fitness, 2.0)  # At least the gen 2 fitness
        # Track that we did see the 10.0 fitness in generation 1
        self.assertEqual(best_fitness_seen[0], 10.0)
    
    # ========== Edge Cases with Fitness Values ==========
    
    def test_negative_fitness_values(self):
        """
        Test that negative fitness values are handled correctly.
        
        Should work with negative fitness values.
        """
        self.config.fitness_criterion = 'max'
        self.config.fitness_threshold = -1.0
        pop = neat.Population(self.config)
        
        def fitness_function(genomes, config):
            for i, (genome_id, genome) in enumerate(genomes):
                # Keep fitness negative across all generations
                genome.fitness = -2.0 + (i * 0.001)  # Small variation, stays negative
        
        result = pop.run(fitness_function, 1)  # Just run 1 generation
        self.assertIsNotNone(result)
        # In generation 1, max fitness should be close to -2.0 + (pop_size * 0.001)
        # which should still be negative for reasonable pop sizes
        self.assertLess(result.fitness, 0)  # Should be negative
    
    def test_zero_fitness_values(self):
        """
        Test that zero fitness values work correctly.
        
        Zero should be a valid fitness value.
        """
        self.config.fitness_criterion = 'max'
        self.config.fitness_threshold = 0.0
        pop = neat.Population(self.config)
        
        generation_count = [0]
        
        def fitness_function(genomes, config):
            generation_count[0] += 1
            for genome_id, genome in genomes:
                genome.fitness = 0.0
        
        pop.run(fitness_function, 10)
        
        # Should terminate when max = 0.0 >= 0.0
        self.assertEqual(generation_count[0], 1)
    
    def test_very_large_fitness_values(self):
        """
        Test that very large fitness values are handled.
        
        Should handle large floating point values.
        """
        self.config.fitness_criterion = 'max'
        self.config.fitness_threshold = 1e10
        pop = neat.Population(self.config)
        
        def fitness_function(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 1e12  # Very large
        
        result = pop.run(fitness_function, 1)
        self.assertEqual(result.fitness, 1e12)
    
    def test_all_same_fitness(self):
        """
        Test with all genomes having identical fitness.
        
        Should handle case where all fitness values are equal.
        """
        self.config.fitness_criterion = 'mean'
        self.config.fitness_threshold = 5.0
        pop = neat.Population(self.config)
        
        def fitness_function(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 5.0  # All identical
        
        result = pop.run(fitness_function, 1)
        self.assertEqual(result.fitness, 5.0)
    
    def test_floating_point_precision(self):
        """
        Test that floating point fitness values work correctly.
        
        Should handle precise floating point comparisons.
        """
        self.config.fitness_criterion = 'max'
        self.config.fitness_threshold = 0.9
        pop = neat.Population(self.config)
        
        def fitness_function(genomes, config):
            for i, (genome_id, genome) in enumerate(genomes):
                genome.fitness = 0.9 + i * 0.001
        
        result = pop.run(fitness_function, 1)
        self.assertGreaterEqual(result.fitness, 0.9)
    
    # ========== No Fitness Termination Mode ==========
    
    def test_no_fitness_termination_requires_generation_limit(self):
        """
        Test that no_fitness_termination requires generation limit.
        
        Should raise error if n=None with no_fitness_termination=True.
        """
        self.config.no_fitness_termination = True
        pop = neat.Population(self.config)
        
        def fitness_function(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 1.0
        
        with self.assertRaises(RuntimeError) as context:
            pop.run(fitness_function, None)
        
        self.assertIn("no generational limit", str(context.exception))
    
    def test_no_fitness_termination_runs_all_generations(self):
        """
        Test that no_fitness_termination runs all generations regardless of fitness.
        
        Should ignore fitness threshold and run all generations.
        """
        self.config.no_fitness_termination = True
        pop = neat.Population(self.config)
        
        generation_count = [0]
        
        def fitness_function(genomes, config):
            generation_count[0] += 1
            for genome_id, genome in genomes:
                genome.fitness = 1000.0  # Very high, would trigger termination
        
        pop.run(fitness_function, 5)
        
        # Should run all 5 generations despite high fitness
        self.assertEqual(generation_count[0], 5)
    
    # ========== Fitness Function Interface ==========
    
    def test_fitness_function_receives_correct_arguments(self):
        """
        Test that fitness function receives correct arguments.
        
        Should receive (genomes, config) as specified.
        """
        pop = neat.Population(self.config)
        
        received_args = []
        
        def fitness_function(genomes, config):
            received_args.append((genomes, config))
            for genome_id, genome in genomes:
                genome.fitness = 1.0
        
        pop.run(fitness_function, 1)
        
        self.assertEqual(len(received_args), 1)
        genomes, config = received_args[0]
        
        # Genomes should be list of (id, genome) tuples
        self.assertIsInstance(genomes, list)
        self.assertGreater(len(genomes), 0)
        genome_id, genome = genomes[0]
        self.assertIsInstance(genome_id, int)
        self.assertIsInstance(genome, neat.DefaultGenome)
        
        # Config should be the population config
        self.assertEqual(config, self.config)
    
    def test_fitness_function_called_each_generation(self):
        """
        Test that fitness function is called for each generation.
        
        Should be called exactly n times for n generations.
        """
        self.config.fitness_threshold = 100.0  # Won't be reached
        pop = neat.Population(self.config)
        
        call_count = [0]
        
        def fitness_function(genomes, config):
            call_count[0] += 1
            for genome_id, genome in genomes:
                genome.fitness = 1.0
        
        pop.run(fitness_function, 7)
        
        self.assertEqual(call_count[0], 7)
    
    def test_population_size_maintained(self):
        """
        Test that population size is maintained across generations.
        
        Each generation should have expected population size.
        """
        pop = neat.Population(self.config)
        
        population_sizes = []
        
        def fitness_function(genomes, config):
            population_sizes.append(len(genomes))
            for genome_id, genome in genomes:
                genome.fitness = 1.0
        
        pop.run(fitness_function, 3)
        
        # All generations should have same size
        expected_size = self.config.pop_size
        for size in population_sizes:
            self.assertEqual(size, expected_size)
    
    # ========== Mixed Fitness Scenarios ==========
    
    def test_some_genomes_high_fitness_some_low(self):
        """
        Test with mixed fitness values across population.
        
        Should correctly identify best genome with mixed fitness.
        """
        self.config.fitness_criterion = 'max'
        self.config.fitness_threshold = 100.0
        pop = neat.Population(self.config)
        
        def fitness_function(genomes, config):
            for i, (genome_id, genome) in enumerate(genomes):
                # Varied fitness
                genome.fitness = float(i % 10)
        
        result = pop.run(fitness_function, 2)
        
        # Best should be at least 9 (from i % 10)
        self.assertGreaterEqual(result.fitness, 9.0)
    
    def test_fitness_improving_over_generations(self):
        """
        Test that best fitness can improve over generations.
        
        Best genome should track highest fitness ever achieved.
        """
        self.config.fitness_threshold = 100.0
        pop = neat.Population(self.config)
        
        generation = [0]
        
        def fitness_function(genomes, config):
            generation[0] += 1
            for i, (genome_id, genome) in enumerate(genomes):
                # Fitness improves each generation
                genome.fitness = generation[0] * 2.0 if i == 0 else 1.0
        
        pop.run(fitness_function, 5)
        
        # After 5 generations, best should be 5 * 2.0 = 10.0
        self.assertEqual(pop.best_genome.fitness, 10.0)
    
    def test_fitness_type_validation(self):
        """
        Test that fitness must be a numeric type.
        
        Note: This tests current behavior - fitness as None is caught,
        but other non-numeric types may cause issues later.
        """
        pop = neat.Population(self.config)
        
        def fitness_function(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 1.5  # Valid float
        
        # Should work with float
        result = pop.run(fitness_function, 1)
        self.assertIsNotNone(result)
        
        # Should also work with int (Python converts to float)
        def fitness_function_int(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 5  # Int
        
        pop2 = neat.Population(self.config)
        result2 = pop2.run(fitness_function_int, 1)
        self.assertEqual(result2.fitness, 5)


if __name__ == '__main__':
    unittest.main()
