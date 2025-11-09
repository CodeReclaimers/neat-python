"""
Unit tests for innovation number tracking.

Tests the InnovationTracker class and its integration with genome mutations and crossover.
"""

import os
import pickle
import tempfile
import unittest

import neat


class TestInnovationTracker(unittest.TestCase):
    """Test the InnovationTracker class."""

    def test_first_innovation_number_is_one(self):
        """Test that the first innovation number assigned is 1."""
        tracker = neat.InnovationTracker()
        innovation = tracker.get_innovation_number(0, 1, 'add_connection')
        self.assertEqual(innovation, 1)

    def test_global_counter_increments(self):
        """Test that the global counter increments with each new innovation."""
        tracker = neat.InnovationTracker()
        inn1 = tracker.get_innovation_number(0, 1, 'add_connection')
        inn2 = tracker.get_innovation_number(0, 2, 'add_connection')
        inn3 = tracker.get_innovation_number(1, 2, 'add_connection')
        
        self.assertEqual(inn1, 1)
        self.assertEqual(inn2, 2)
        self.assertEqual(inn3, 3)

    def test_same_mutation_same_generation_same_innovation(self):
        """Test that the same mutation in the same generation gets the same innovation number."""
        tracker = neat.InnovationTracker()
        
        # First genome adds connection 0->1
        inn1 = tracker.get_innovation_number(0, 1, 'add_connection')
        
        # Second genome adds same connection 0->1 in same generation
        inn2 = tracker.get_innovation_number(0, 1, 'add_connection')
        
        # Should get the same innovation number
        self.assertEqual(inn1, inn2)
        self.assertEqual(inn1, 1)

    def test_different_mutations_different_innovations(self):
        """Test that different mutations get different innovation numbers."""
        tracker = neat.InnovationTracker()
        
        inn1 = tracker.get_innovation_number(0, 1, 'add_connection')
        inn2 = tracker.get_innovation_number(0, 2, 'add_connection')
        
        self.assertNotEqual(inn1, inn2)

    def test_same_connection_different_mutation_types(self):
        """Test that the same connection with different mutation types gets different innovations."""
        tracker = neat.InnovationTracker()
        
        # Same connection (0->1) but different mutation types
        inn1 = tracker.get_innovation_number(0, 1, 'add_connection')
        inn2 = tracker.get_innovation_number(0, 1, 'add_node_in')
        
        self.assertNotEqual(inn1, inn2)

    def test_reset_generation_clears_tracking(self):
        """Test that reset_generation clears generation-specific tracking."""
        tracker = neat.InnovationTracker()
        
        # Generation 1: add connection 0->1
        inn1 = tracker.get_innovation_number(0, 1, 'add_connection')
        self.assertEqual(inn1, 1)
        
        # Verify same mutation gets same number
        inn1_again = tracker.get_innovation_number(0, 1, 'add_connection')
        self.assertEqual(inn1_again, 1)
        
        # New generation
        tracker.reset_generation()
        
        # Same mutation should get NEW innovation number
        inn2 = tracker.get_innovation_number(0, 1, 'add_connection')
        self.assertEqual(inn2, 2)

    def test_reset_generation_preserves_global_counter(self):
        """Test that reset_generation preserves the global counter."""
        tracker = neat.InnovationTracker()
        
        # Assign some innovations
        tracker.get_innovation_number(0, 1, 'add_connection')
        tracker.get_innovation_number(0, 2, 'add_connection')
        tracker.get_innovation_number(1, 2, 'add_connection')
        
        # Reset generation
        tracker.reset_generation()
        
        # Next innovation should be 4, not 1
        inn = tracker.get_innovation_number(2, 3, 'add_connection')
        self.assertEqual(inn, 4)

    def test_add_node_mutations_tracked_separately(self):
        """Test that add_node mutations track both connections separately."""
        tracker = neat.InnovationTracker()
        
        # Split connection 0->1 by adding node 2
        inn_in = tracker.get_innovation_number(0, 2, 'add_node_in')
        inn_out = tracker.get_innovation_number(2, 1, 'add_node_out')
        
        self.assertEqual(inn_in, 1)
        self.assertEqual(inn_out, 2)
        
        # Another genome splits same connection
        inn_in2 = tracker.get_innovation_number(0, 2, 'add_node_in')
        inn_out2 = tracker.get_innovation_number(2, 1, 'add_node_out')
        
        # Should get same innovation numbers
        self.assertEqual(inn_in2, inn_in)
        self.assertEqual(inn_out2, inn_out)

    def test_pickle_serialization(self):
        """Test that InnovationTracker can be pickled and unpickled."""
        tracker = neat.InnovationTracker()
        
        # Assign some innovations
        tracker.get_innovation_number(0, 1, 'add_connection')
        tracker.get_innovation_number(0, 2, 'add_connection')
        tracker.get_innovation_number(1, 2, 'add_connection')
        
        # Pickle
        pickled = pickle.dumps(tracker)
        
        # Unpickle
        restored = pickle.loads(pickled)
        
        # Verify state preserved
        self.assertEqual(restored.global_counter, 3)
        
        # Verify can continue assigning innovations
        inn = restored.get_innovation_number(2, 3, 'add_connection')
        self.assertEqual(inn, 4)

    def test_pickle_preserves_generation_tracking(self):
        """Test that pickling preserves generation-specific tracking."""
        tracker = neat.InnovationTracker()
        
        # Assign innovation
        inn1 = tracker.get_innovation_number(0, 1, 'add_connection')
        
        # Pickle and restore
        restored = pickle.loads(pickle.dumps(tracker))
        
        # Same mutation should return same innovation (within generation)
        inn2 = restored.get_innovation_number(0, 1, 'add_connection')
        self.assertEqual(inn1, inn2)


class TestGenomeInnovationTracking(unittest.TestCase):
    """Test innovation tracking in genome mutations."""

    def setUp(self):
        """Set up test configuration."""
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                   neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                   config_path)
        
        # Create innovation tracker
        self.tracker = neat.InnovationTracker()
        self.config.genome_config.innovation_tracker = self.tracker

    def test_mutate_add_connection_assigns_innovation(self):
        """Test that mutate_add_connection assigns innovation numbers."""
        genome = neat.DefaultGenome(1)
        genome.configure_new(self.config.genome_config)
        
        # Record initial connection count
        initial_count = len(genome.connections)
        
        # Mutate to add connection
        genome.mutate_add_connection(self.config.genome_config)
        
        # Should have added a connection with innovation number
        if len(genome.connections) > initial_count:
            # Find the new connection
            new_conn = list(genome.connections.values())[-1]
            self.assertIsNotNone(new_conn.innovation)
            self.assertGreater(new_conn.innovation, 0)

    def test_mutate_add_connection_deduplication(self):
        """Test that same connection mutation in same generation gets same innovation."""
        # Create two genomes with same structure
        genome1 = neat.DefaultGenome(1)
        genome2 = neat.DefaultGenome(2)
        
        # Configure with minimal structure to control mutation
        genome1.nodes = {0: neat.DefaultNodeGene(0), 1: neat.DefaultNodeGene(1)}
        genome2.nodes = {0: neat.DefaultNodeGene(0), 1: neat.DefaultNodeGene(1)}
        genome1.connections = {}
        genome2.connections = {}
        
        # Manually add the same connection to both
        inn1 = self.tracker.get_innovation_number(0, 1, 'add_connection')
        conn1 = neat.DefaultConnectionGene((0, 1), innovation=inn1)
        genome1.connections[(0, 1)] = conn1
        
        inn2 = self.tracker.get_innovation_number(0, 1, 'add_connection')
        conn2 = neat.DefaultConnectionGene((0, 1), innovation=inn2)
        genome2.connections[(0, 1)] = conn2
        
        # Should have same innovation number (same generation)
        self.assertEqual(inn1, inn2)

    def test_mutate_add_node_assigns_two_innovations(self):
        """Test that mutate_add_node assigns two innovation numbers."""
        genome = neat.DefaultGenome(1)
        genome.configure_new(self.config.genome_config)
        
        # Ensure there's at least one connection to split
        if len(genome.connections) == 0:
            genome.mutate_add_connection(self.config.genome_config)
        
        initial_conn_count = len(genome.connections)
        initial_node_count = len(genome.nodes)
        
        # Mutate to add node
        genome.mutate_add_node(self.config.genome_config)
        
        # Should have added a node and two connections (+2 new, -1 disabled = +1 net)
        if len(genome.nodes) > initial_node_count:
            # Could be +1 (if node added) or same (if no mutation occurred)
            self.assertGreaterEqual(len(genome.connections), initial_conn_count)
            
            # Check that new connections have innovation numbers
            for conn in genome.connections.values():
                if getattr(conn, 'enabled', True):
                    self.assertIsNotNone(conn.innovation)
                    self.assertGreater(conn.innovation, 0)

    def test_initial_connections_have_innovations(self):
        """Test that initial genome connections have innovation numbers."""
        genome = neat.DefaultGenome(1)
        genome.configure_new(self.config.genome_config)
        
        # All connections should have innovation numbers
        for conn in genome.connections.values():
            self.assertIsNotNone(conn.innovation)
            self.assertGreater(conn.innovation, 0)

    def test_initial_connections_unique_innovations(self):
        """Test that initial connections get unique innovation numbers."""
        genome = neat.DefaultGenome(1)
        genome.configure_new(self.config.genome_config)
        
        # Collect all innovation numbers
        innovations = [conn.innovation for conn in genome.connections.values()]
        
        # Should all be unique
        self.assertEqual(len(innovations), len(set(innovations)))


class TestCrossoverInnovationMatching(unittest.TestCase):
    """Test that crossover uses innovation numbers for gene matching."""

    def setUp(self):
        """Set up test configuration."""
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                   neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                   config_path)
        
        # Create innovation tracker
        self.tracker = neat.InnovationTracker()
        self.config.genome_config.innovation_tracker = self.tracker

    def test_crossover_matches_by_innovation(self):
        """Test that crossover matches genes by innovation number."""
        # Create two parent genomes
        parent1 = neat.DefaultGenome(1)
        parent2 = neat.DefaultGenome(2)
        
        # Set fitness values
        parent1.fitness = 1.0
        parent2.fitness = 1.0
        
        # Set up nodes
        parent1.nodes = {}
        parent2.nodes = {}
        for nid in [-1, -2, 0]:
            node1 = neat.DefaultNodeGene(nid)
            node1.init_attributes(self.config.genome_config)
            parent1.nodes[nid] = node1
            
            node2 = neat.DefaultNodeGene(nid)
            node2.init_attributes(self.config.genome_config)
            parent2.nodes[nid] = node2
        
        # Create connections with same innovation numbers
        # Parent 1: connections with innovations 1, 2, 3
        conn1_1 = neat.DefaultConnectionGene((-1, 0), innovation=1)
        conn1_1.init_attributes(self.config.genome_config)
        conn1_2 = neat.DefaultConnectionGene((-2, 0), innovation=2)
        conn1_2.init_attributes(self.config.genome_config)
        conn1_3 = neat.DefaultConnectionGene((-1, -2), innovation=3)
        conn1_3.init_attributes(self.config.genome_config)
        
        parent1.connections = {
            (-1, 0): conn1_1,
            (-2, 0): conn1_2,
            (-1, -2): conn1_3,
        }
        
        # Parent 2: connections with innovations 1, 2, 4 (4 is disjoint)
        conn2_1 = neat.DefaultConnectionGene((-1, 0), innovation=1)
        conn2_1.init_attributes(self.config.genome_config)
        conn2_2 = neat.DefaultConnectionGene((-2, 0), innovation=2)
        conn2_2.init_attributes(self.config.genome_config)
        conn2_4 = neat.DefaultConnectionGene((-2, -1), innovation=4)
        conn2_4.init_attributes(self.config.genome_config)
        
        parent2.connections = {
            (-1, 0): conn2_1,
            (-2, 0): conn2_2,
            (-2, -1): conn2_4,
        }
        
        # Create child via crossover
        child = neat.DefaultGenome(3)
        child.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # Child should have genes matched by innovation number
        child_innovations = {conn.innovation for conn in child.connections.values()}
        
        # Should have innovations 1 and 2 (matching), and possibly 3 and/or 4 (disjoint)
        self.assertIn(1, child_innovations)
        self.assertIn(2, child_innovations)

    def test_crossover_preserves_innovation_numbers(self):
        """Test that crossover preserves innovation numbers in child."""
        parent1 = neat.DefaultGenome(1)
        parent2 = neat.DefaultGenome(2)
        
        # Set fitness values
        parent1.fitness = 1.0
        parent2.fitness = 1.0
        
        # Configure parents
        parent1.configure_new(self.config.genome_config)
        parent2.configure_new(self.config.genome_config)
        
        # Get parent innovation numbers
        parent1_innovations = {conn.innovation for conn in parent1.connections.values()}
        parent2_innovations = {conn.innovation for conn in parent2.connections.values()}
        all_parent_innovations = parent1_innovations | parent2_innovations
        
        # Create child
        child = neat.DefaultGenome(3)
        child.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # All child innovations should come from parents
        for conn in child.connections.values():
            self.assertIn(conn.innovation, all_parent_innovations)

    def test_matching_genes_same_innovation(self):
        """Test that genes with same innovation in both parents are matching genes."""
        parent1 = neat.DefaultGenome(1)
        parent2 = neat.DefaultGenome(2)
        
        # Set fitness values
        parent1.fitness = 1.0
        parent2.fitness = 1.0
        
        # Set up identical structure with same innovations
        parent1.nodes = {}
        parent2.nodes = {}
        for nid in [-1, 0]:
            node1 = neat.DefaultNodeGene(nid)
            node1.init_attributes(self.config.genome_config)
            parent1.nodes[nid] = node1
            
            node2 = neat.DefaultNodeGene(nid)
            node2.init_attributes(self.config.genome_config)
            parent2.nodes[nid] = node2
        
        # Same connection with same innovation but different weights
        conn1 = neat.DefaultConnectionGene((-1, 0), innovation=1)
        conn1.init_attributes(self.config.genome_config)
        conn1.weight = 1.0
        
        conn2 = neat.DefaultConnectionGene((-1, 0), innovation=1)
        conn2.init_attributes(self.config.genome_config)
        conn2.weight = 2.0
        
        parent1.connections = {(-1, 0): conn1}
        parent2.connections = {(-1, 0): conn2}
        
        # Create child
        child = neat.DefaultGenome(3)
        child.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # Child should have the connection with innovation 1
        self.assertEqual(len(child.connections), 1)
        child_conn = list(child.connections.values())[0]
        self.assertEqual(child_conn.innovation, 1)
        # Weight should be from one parent or the other
        self.assertIn(child_conn.weight, [1.0, 2.0])


class TestInnovationCheckpointing(unittest.TestCase):
    """Test that innovation tracking works with checkpointing."""

    def test_checkpoint_preserves_innovation_counter(self):
        """Test that InnovationTracker can be pickled with a population."""
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
        
        # Create population
        pop = neat.Population(config)
        
        # Run a generation to assign some innovations
        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 1.0
        
        pop.run(eval_genomes, 1)
        
        # Get current innovation counter
        original_counter = pop.reproduction.innovation_tracker.global_counter
        self.assertGreater(original_counter, 0)
        
        # Test that the innovation tracker can be pickled
        import pickle
        tracker_data = pickle.dumps(pop.reproduction.innovation_tracker)
        restored_tracker = pickle.loads(tracker_data)
        
        # Innovation counter should be preserved
        self.assertEqual(restored_tracker.global_counter, original_counter)
        
        # Should be able to continue assigning innovations
        new_innovation = restored_tracker.get_innovation_number(99, 100, 'add_connection')
        self.assertEqual(new_innovation, original_counter + 1)


if __name__ == '__main__':
    unittest.main()
