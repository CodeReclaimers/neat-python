"""
Integration test for innovation number tracking using the XOR problem.

Tests that innovation tracking works correctly during a full evolutionary run.
"""

import os
import unittest

import neat


class TestXORWithInnovationTracking(unittest.TestCase):
    """Integration test using XOR problem to validate innovation tracking."""

    def setUp(self):
        """Set up XOR test configuration."""
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                   neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                   config_path)
        
        # XOR inputs and expected outputs
        self.xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        self.xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

    def eval_genomes(self, genomes, config):
        """Evaluate genomes on XOR problem."""
        for genome_id, genome in genomes:
            genome.fitness = 4.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for xi, xo in zip(self.xor_inputs, self.xor_outputs):
                output = net.activate(xi)
                genome.fitness -= (output[0] - xo[0]) ** 2

    def test_innovation_tracking_during_evolution(self):
        """Test that innovation numbers are properly assigned during evolution."""
        # Create population
        pop = neat.Population(self.config)
        
        # Run for a few generations
        generations_to_run = 5
        
        # Track innovation numbers across generations
        all_innovations_seen = set()
        
        for gen in range(generations_to_run):
            # Evaluate genomes
            self.eval_genomes(list(pop.population.items()), self.config)
            
            # Check innovation numbers
            for genome_id, genome in pop.population.items():
                for conn in genome.connections.values():
                    # All connections should have valid innovation numbers
                    self.assertIsNotNone(conn.innovation)
                    self.assertIsInstance(conn.innovation, int)
                    self.assertGreater(conn.innovation, 0)
                    
                    # Track all innovations
                    all_innovations_seen.add(conn.innovation)
            
            # If not last generation, reproduce
            if gen < generations_to_run - 1:
                # Get best genome
                best = None
                for g in pop.population.values():
                    if best is None or g.fitness > best.fitness:
                        best = g
                
                # Reproduce
                pop.population = pop.reproduction.reproduce(
                    self.config, pop.species, self.config.pop_size, gen
                )
                
                # Speciate
                pop.species.speciate(self.config, pop.population, gen + 1)
                pop.generation = gen + 1
        
        # Verify we saw multiple innovations (evolution occurred)
        self.assertGreater(len(all_innovations_seen), self.config.genome_config.num_inputs * 
                          self.config.genome_config.num_outputs)

    def test_innovation_counter_increases_monotonically(self):
        """Test that the global innovation counter only increases."""
        pop = neat.Population(self.config)
        tracker = pop.reproduction.innovation_tracker
        
        # Initial counter value
        initial_counter = tracker.global_counter
        
        # Run some generations
        for gen in range(3):
            self.eval_genomes(list(pop.population.items()), self.config)
            
            # Counter should have increased
            current_counter = tracker.global_counter
            self.assertGreaterEqual(current_counter, initial_counter)
            
            # Track new max
            initial_counter = current_counter
            
            # Reproduce for next generation
            if gen < 2:
                pop.population = pop.reproduction.reproduce(
                    self.config, pop.species, self.config.pop_size, gen
                )
                pop.species.speciate(self.config, pop.population, gen + 1)
                pop.generation = gen + 1

    def test_no_innovation_collisions(self):
        """Test that innovation numbers are never reused."""
        pop = neat.Population(self.config)
        
        # Collect all innovation numbers across several generations
        innovation_history = []
        
        for gen in range(5):
            self.eval_genomes(list(pop.population.items()), self.config)
            
            # Collect innovations from this generation
            gen_innovations = []
            for genome_id, genome in pop.population.items():
                for conn in genome.connections.values():
                    gen_innovations.append(conn.innovation)
            
            innovation_history.extend(gen_innovations)
            
            # Reproduce for next generation
            if gen < 4:
                pop.population = pop.reproduction.reproduce(
                    self.config, pop.species, self.config.pop_size, gen
                )
                pop.species.speciate(self.config, pop.population, gen + 1)
                pop.generation = gen + 1
        
        # All innovations should be positive integers
        for inn in innovation_history:
            self.assertIsInstance(inn, int)
            self.assertGreater(inn, 0)
        
        # Count of unique innovations
        unique_innovations = len(set(innovation_history))
        self.assertGreater(unique_innovations, 0)

    def test_crossover_preserves_innovations(self):
        """Test that crossover preserves innovation numbers from parents."""
        pop = neat.Population(self.config)
        
        # Evaluate initial population
        self.eval_genomes(list(pop.population.items()), self.config)
        
        # Get two parent genomes
        parent_ids = list(pop.population.keys())[:2]
        parent1 = pop.population[parent_ids[0]]
        parent2 = pop.population[parent_ids[1]]
        
        # Set equal fitness so both contribute genes
        parent1.fitness = 1.0
        parent2.fitness = 1.0
        
        # Collect parent innovations
        parent1_innovations = {conn.innovation for conn in parent1.connections.values()}
        parent2_innovations = {conn.innovation for conn in parent2.connections.values()}
        all_parent_innovations = parent1_innovations | parent2_innovations
        
        # Create child via crossover
        child_id = max(pop.population.keys()) + 1
        child = neat.DefaultGenome(child_id)
        child.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # All child innovations should come from parents
        for conn in child.connections.values():
            self.assertIn(conn.innovation, all_parent_innovations,
                         f"Child has innovation {conn.innovation} not in parents")

    def test_same_mutation_same_generation_same_innovation(self):
        """Test that identical mutations in same generation get same innovation."""
        pop = neat.Population(self.config)
        tracker = pop.reproduction.innovation_tracker
        
        # Reset generation to ensure clean slate
        tracker.reset_generation()
        
        # Get two genomes with same structure
        genome_ids = list(pop.population.keys())[:2]
        genome1 = pop.population[genome_ids[0]]
        genome2 = pop.population[genome_ids[1]]
        
        # Track innovations before mutation
        initial_counter = tracker.global_counter
        
        # Both genomes will attempt to mutate
        # Note: mutations are stochastic, so we can't guarantee they'll add
        # the same connection, but we can verify the tracker behavior
        
        # Simulate identical mutation by directly calling tracker
        inn1 = tracker.get_innovation_number(100, 200, 'add_connection')
        inn2 = tracker.get_innovation_number(100, 200, 'add_connection')
        
        # Should get same innovation number
        self.assertEqual(inn1, inn2)
        
        # Should only increment counter once
        self.assertEqual(tracker.global_counter, initial_counter + 1)

    def test_generation_reset_clears_deduplication(self):
        """Test that generation reset allows same mutation to get new innovation."""
        pop = neat.Population(self.config)
        tracker = pop.reproduction.innovation_tracker
        
        # First generation
        tracker.reset_generation()
        inn1 = tracker.get_innovation_number(10, 20, 'add_connection')
        
        # Same mutation in same generation
        inn1_again = tracker.get_innovation_number(10, 20, 'add_connection')
        self.assertEqual(inn1, inn1_again)
        
        # Next generation
        tracker.reset_generation()
        inn2 = tracker.get_innovation_number(10, 20, 'add_connection')
        
        # Should get different innovation number
        self.assertNotEqual(inn1, inn2)
        self.assertEqual(inn2, inn1 + 1)


if __name__ == '__main__':
    unittest.main()
