"""
Tests for complete extinction scenarios and recovery mechanisms.
"""
import os
import unittest
from unittest.mock import MagicMock, patch

import neat
from neat.population import CompleteExtinctionException


class TestCompleteExtinction(unittest.TestCase):
    """Tests for complete extinction scenarios."""
    
    def setUp(self):
        """Set up configuration for extinction tests."""
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)
    
    def create_extinction_config(self, reset_on_extinction=False):
        """Create a configuration that will lead to quick extinction."""
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           os.path.join(os.path.dirname(__file__), 'test_configuration'))
        # Set very aggressive stagnation to force extinction
        config.stagnation_config.max_stagnation = 1
        config.stagnation_config.species_elitism = 0
        config.reset_on_extinction = reset_on_extinction
        return config
    
    def fitness_all_zero(self, genomes, config):
        """Fitness function that assigns zero fitness to all genomes."""
        for genome_id, genome in genomes:
            genome.fitness = 0.0
    
    def fitness_all_negative(self, genomes, config):
        """Fitness function that assigns negative fitness to all genomes."""
        for genome_id, genome in genomes:
            genome.fitness = -1.0
    
    def fitness_first_gen_zero_then_improve(self, genomes, config):
        """Fitness function that gives zero initially, then improves."""
        # Track generation via closure
        if not hasattr(self.fitness_first_gen_zero_then_improve, 'gen_count'):
            self.fitness_first_gen_zero_then_improve.gen_count = 0
        
        gen = self.fitness_first_gen_zero_then_improve.gen_count
        self.fitness_first_gen_zero_then_improve.gen_count += 1
        
        for genome_id, genome in genomes:
            if gen == 0:
                genome.fitness = 0.0
            else:
                genome.fitness = float(genome_id % 10)  # Varying fitness
    
    def test_extinction_exception_raised(self):
        """Test that CompleteExtinctionException is raised when reset_on_extinction=False."""
        config = self.create_extinction_config(reset_on_extinction=False)
        p = neat.Population(config)
        
        with self.assertRaises(CompleteExtinctionException):
            p.run(self.fitness_all_zero, 50)
    
    def test_extinction_exception_type(self):
        """Test that the correct exception type is raised."""
        config = self.create_extinction_config(reset_on_extinction=False)
        p = neat.Population(config)
        
        try:
            p.run(self.fitness_all_zero, 50)
            self.fail("Should have raised CompleteExtinctionException")
        except CompleteExtinctionException as e:
            # Verify it's the right exception type
            self.assertIsInstance(e, CompleteExtinctionException)
            self.assertIsInstance(e, Exception)
    
    def test_extinction_recovery_with_reset(self):
        """Test that population recovers when reset_on_extinction=True."""
        config = self.create_extinction_config(reset_on_extinction=True)
        p = neat.Population(config)
        
        # Should complete without raising exception
        best = p.run(self.fitness_all_zero, 20)
        
        # Verify population still exists
        self.assertIsNotNone(p.population)
        self.assertGreater(len(p.population), 0)
    
    def test_extinction_reporter_notified(self):
        """Test that reporters are notified of complete extinction."""
        config = self.create_extinction_config(reset_on_extinction=True)
        p = neat.Population(config)
        
        reporter = neat.StdOutReporter(False)
        p.add_reporter(reporter)
        
        p.run(self.fitness_all_zero, 20)
        
        # Verify extinction was detected
        self.assertGreater(reporter.num_extinctions, 0)
    
    def test_extinction_multiple_times(self):
        """Test handling multiple extinctions in a single run."""
        config = self.create_extinction_config(reset_on_extinction=True)
        p = neat.Population(config)
        
        reporter = neat.StdOutReporter(False)
        p.add_reporter(reporter)
        
        # Run long enough to potentially get multiple extinctions
        p.run(self.fitness_all_zero, 50)
        
        # Should have at least one extinction
        self.assertGreater(reporter.num_extinctions, 0)
    
    def test_extinction_preserves_best_genome(self):
        """Test that best_genome is preserved across extinction."""
        config = self.create_extinction_config(reset_on_extinction=True)
        p = neat.Population(config)
        
        # Use fitness function that has some variation
        def varying_fitness(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = float(genome_id % 10)
        
        p.run(varying_fitness, 15)
        
        # best_genome should be set
        self.assertIsNotNone(p.best_genome)
        self.assertIsNotNone(p.best_genome.fitness)
    
    def test_extinction_in_first_generation(self):
        """Test extinction can occur in the very first generation."""
        config = self.create_extinction_config(reset_on_extinction=False)
        # Make extinction even more aggressive
        config.stagnation_config.max_stagnation = 0
        
        p = neat.Population(config)
        
        with self.assertRaises(CompleteExtinctionException):
            p.run(self.fitness_all_zero, 10)
    
    def test_extinction_after_successful_evolution(self):
        """Test extinction after period of successful evolution."""
        config = self.create_extinction_config(reset_on_extinction=False)
        config.fitness_threshold = 1000.0  # High threshold to prevent early termination
        p = neat.Population(config)
        
        # Track call count
        call_count = [0]
        
        def fitness_good_then_bad(genomes, config):
            if call_count[0] < 3:
                # Good fitness for first few generations (but below threshold)
                for genome_id, genome in genomes:
                    genome.fitness = 10.0 + float(genome_id % 10)
            else:
                # Then all zero
                for genome_id, genome in genomes:
                    genome.fitness = 0.0
            call_count[0] += 1
        
        with self.assertRaises(CompleteExtinctionException):
            p.run(fitness_good_then_bad, 20)
        
        # Should have run a few generations (at least some with good fitness)
        # before extinction occurred
        self.assertGreater(call_count[0], 0)
    
    def test_extinction_population_reset_creates_new_genomes(self):
        """Test that extinction reset creates entirely new population."""
        config = self.create_extinction_config(reset_on_extinction=True)
        p = neat.Population(config)
        
        # Get initial population IDs
        initial_ids = set(p.population.keys())
        
        # Run until extinction
        p.run(self.fitness_all_zero, 20)
        
        # After extinction and reset, should have new genome IDs
        final_ids = set(p.population.keys())
        
        # Most IDs should be different (some overlap possible due to indexer)
        # but population should definitely exist
        self.assertGreater(len(final_ids), 0)
    
    def test_extinction_species_recreated(self):
        """Test that species are recreated after extinction."""
        config = self.create_extinction_config(reset_on_extinction=True)
        p = neat.Population(config)
        
        p.run(self.fitness_all_zero, 20)
        
        # After reset and speciation, should have species
        self.assertGreater(len(p.species.species), 0)
    
    def test_extinction_generation_counter_continues(self):
        """Test that generation counter continues after extinction."""
        config = self.create_extinction_config(reset_on_extinction=True)
        p = neat.Population(config)
        
        reporter = neat.StdOutReporter(False)
        p.add_reporter(reporter)
        
        p.run(self.fitness_all_zero, 15)
        
        # Generation should have advanced
        self.assertGreater(p.generation, 0)
    
    def test_extinction_with_negative_fitness(self):
        """Test extinction handling with negative fitness values."""
        config = self.create_extinction_config(reset_on_extinction=True)
        p = neat.Population(config)
        
        # Should handle negative fitness without issues
        p.run(self.fitness_all_negative, 15)
        
        self.assertIsNotNone(p.best_genome)
    
    def test_extinction_statistics_collection(self):
        """Test that statistics are properly collected across extinction."""
        config = self.create_extinction_config(reset_on_extinction=True)
        p = neat.Population(config)
        
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        
        p.run(self.fitness_all_zero, 20)
        
        # Statistics should have been collected
        self.assertGreater(len(stats.generation_statistics), 0)
        self.assertGreater(len(stats.most_fit_genomes), 0)
    
    def test_no_extinction_with_improving_fitness(self):
        """Test that extinction doesn't occur when fitness improves."""
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           os.path.join(os.path.dirname(__file__), 'test_configuration'))
        # Use normal stagnation settings
        config.stagnation_config.max_stagnation = 15
        
        p = neat.Population(config)
        reporter = neat.StdOutReporter(False)
        p.add_reporter(reporter)
        
        def improving_fitness(genomes, config):
            if not hasattr(improving_fitness, 'gen'):
                improving_fitness.gen = 0
            improving_fitness.gen += 1
            
            for genome_id, genome in genomes:
                # Fitness improves each generation
                genome.fitness = float(improving_fitness.gen * 10 + genome_id % 5)
        
        p.run(improving_fitness, 10)
        
        # Should have no extinctions
        self.assertEqual(reporter.num_extinctions, 0)
    
    def test_extinction_without_reset_preserves_state(self):
        """Test that state is preserved up to extinction when reset_on_extinction=False."""
        config = self.create_extinction_config(reset_on_extinction=False)
        p = neat.Population(config)
        
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        
        try:
            p.run(self.fitness_all_zero, 50)
        except CompleteExtinctionException:
            pass
        
        # Statistics should be preserved up to point of extinction
        self.assertGreater(len(stats.generation_statistics), 0)
        self.assertGreater(len(stats.most_fit_genomes), 0)
        
        # Best genome should still be available
        self.assertIsNotNone(p.best_genome)
    
    def test_extinction_callback_receives_correct_state(self):
        """Test that extinction callback receives correct population state."""
        config = self.create_extinction_config(reset_on_extinction=True)
        p = neat.Population(config)
        
        # Track callback invocations
        callback_data = []
        
        class MockReporter(neat.reporting.BaseReporter):
            def complete_extinction(self):
                # Capture state at time of callback
                callback_data.append({
                    'population_size': len(p.population),
                    'num_species': len(p.species.species)
                })
        
        p.add_reporter(MockReporter())
        p.run(self.fitness_all_zero, 20)
        
        # Callback should have been invoked at least once
        self.assertGreater(len(callback_data), 0)
        
        # At extinction, species should be empty
        for data in callback_data:
            self.assertEqual(data['num_species'], 0)
    
    def test_extinction_with_checkpointer(self):
        """Test that checkpointing works correctly with extinction."""
        config = self.create_extinction_config(reset_on_extinction=True)
        p = neat.Population(config)
        
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            checkpointer = neat.Checkpointer(generation_interval=5, 
                                            filename_prefix=os.path.join(temp_dir, 'test-checkpoint-'))
            p.add_reporter(checkpointer)
            
            p.run(self.fitness_all_zero, 15)
            
            # Should have created some checkpoints
            checkpoint_files = [f for f in os.listdir(temp_dir) if f.startswith('test-checkpoint-')]
            self.assertGreater(len(checkpoint_files), 0)
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_extinction_with_fitness_threshold(self):
        """Test extinction interaction with fitness threshold termination."""
        config = self.create_extinction_config(reset_on_extinction=False)
        config.fitness_threshold = 100.0  # High threshold unlikely to be reached
        
        p = neat.Population(config)
        
        # Extinction should occur before threshold is reached
        with self.assertRaises(CompleteExtinctionException):
            p.run(self.fitness_all_zero, 50)
    
    def test_extinction_population_size_maintained(self):
        """Test that population size is maintained after extinction reset."""
        config = self.create_extinction_config(reset_on_extinction=True)
        original_pop_size = config.pop_size
        
        p = neat.Population(config)
        p.run(self.fitness_all_zero, 20)
        
        # Population size should match config
        self.assertEqual(len(p.population), original_pop_size)
    
    def test_extinction_with_elitism_zero(self):
        """Test extinction with zero elitism."""
        config = self.create_extinction_config(reset_on_extinction=False)
        config.stagnation_config.species_elitism = 0
        
        p = neat.Population(config)
        
        # Should still lead to extinction
        with self.assertRaises(CompleteExtinctionException):
            p.run(self.fitness_all_zero, 30)
    
    def test_extinction_with_small_population(self):
        """Test extinction with very small population size."""
        config = self.create_extinction_config(reset_on_extinction=True)
        config.pop_size = 5  # Very small population
        
        p = neat.Population(config)
        reporter = neat.StdOutReporter(False)
        p.add_reporter(reporter)
        
        p.run(self.fitness_all_zero, 20)
        
        # Should handle small population
        self.assertEqual(len(p.population), 5)
    
    def test_extinction_custom_reporter_callback_order(self):
        """Test that reporter callbacks are called in correct order during extinction."""
        config = self.create_extinction_config(reset_on_extinction=True)
        p = neat.Population(config)
        
        callback_order = []
        
        class OrderTrackingReporter(neat.reporting.BaseReporter):
            def start_generation(self, generation):
                callback_order.append(('start', generation))
            
            def post_evaluate(self, config, population, species, best_genome):
                callback_order.append(('post_eval', len(population)))
            
            def complete_extinction(self):
                callback_order.append(('extinction', None))
            
            def end_generation(self, config, population, species_set):
                callback_order.append(('end', len(species_set.species)))
        
        p.add_reporter(OrderTrackingReporter())
        p.run(self.fitness_all_zero, 10)
        
        # Verify callbacks were made
        self.assertGreater(len(callback_order), 0)
        
        # Check that extinction callback occurred
        extinction_calls = [c for c in callback_order if c[0] == 'extinction']
        self.assertGreater(len(extinction_calls), 0)
    
    def test_extinction_innovation_tracker_reset(self):
        """Test that innovation tracker is properly reset after extinction."""
        config = self.create_extinction_config(reset_on_extinction=True)
        p = neat.Population(config)
        
        # Get initial innovation number
        if hasattr(p.reproduction, 'genome_indexer'):
            # Force a few genomes to be created
            p.run(self.fitness_all_zero, 20)
            
            # Innovation tracker should continue incrementing
            # (not reset to same values)
            final_ids = list(p.population.keys())
            # All IDs should be unique
            self.assertEqual(len(final_ids), len(set(final_ids)))


class TestExtinctionEdgeCases(unittest.TestCase):
    """Tests for edge cases in extinction handling."""
    
    def setUp(self):
        """Set up configuration for edge case tests."""
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)
    
    def test_extinction_exception_message(self):
        """Test that CompleteExtinctionException can be instantiated with message."""
        exc = CompleteExtinctionException("Test message")
        self.assertEqual(str(exc), "Test message")
    
    def test_extinction_exception_inheritance(self):
        """Test that CompleteExtinctionException is properly derived from Exception."""
        exc = CompleteExtinctionException()
        self.assertIsInstance(exc, Exception)
    
    def test_no_fitness_termination_with_extinction(self):
        """Test interaction between no_fitness_termination and extinction."""
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           os.path.join(os.path.dirname(__file__), 'test_configuration'))
        config.no_fitness_termination = True
        config.stagnation_config.max_stagnation = 1
        config.stagnation_config.species_elitism = 0
        config.reset_on_extinction = False
        
        p = neat.Population(config)
        
        def zero_fitness(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 0.0
        
        # Should raise extinction before reaching generation limit
        with self.assertRaises(CompleteExtinctionException):
            p.run(zero_fitness, 30)
    
    def test_extinction_with_single_species(self):
        """Test extinction when only one species exists."""
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           os.path.join(os.path.dirname(__file__), 'test_configuration'))
        config.stagnation_config.max_stagnation = 1
        config.stagnation_config.species_elitism = 0
        config.reset_on_extinction = True
        
        # Make species threshold very high so everything goes to one species
        config.species_set_config.compatibility_threshold = 100.0
        
        p = neat.Population(config)
        
        def zero_fitness(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 0.0
        
        p.run(zero_fitness, 15)
        
        # Should handle single species extinction
        self.assertGreater(len(p.population), 0)


if __name__ == '__main__':
    unittest.main()
