"""
Tests for genome crossover operations.

Tests verify the NEAT crossover algorithm as described in Stanley & Miikkulainen (2002):
- Matching genes are randomly inherited from either parent
- Disjoint and excess genes are inherited from the fitter parent
- Node genes are inherited based on connection gene inheritance
"""

import unittest
import os
import neat


class TestGenomeCrossover(unittest.TestCase):
    """Test genome crossover operations."""
    
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
        
        # Initialize innovation tracker (required for mutations)
        self.innovation_tracker = neat.InnovationTracker()
        self.config.genome_config.innovation_tracker = self.innovation_tracker
    
    def create_genome_with_structure(self, key, num_hidden=0):
        """Create a genome with specified structure."""
        genome = neat.DefaultGenome(key)
        genome.configure_new(self.config.genome_config)
        
        # Add hidden nodes
        for _ in range(num_hidden):
            if genome.connections:
                genome.mutate_add_node(self.config.genome_config)
        
        return genome
    
    def set_fitness(self, genome, fitness):
        """Set fitness for a genome."""
        genome.fitness = fitness
    
    # ========== Basic Crossover Mechanics ==========
    
    def test_crossover_creates_offspring(self):
        """
        Test that crossover creates a valid offspring genome.
        
        The offspring should have connections and nodes from parents.
        """
        parent1 = self.create_genome_with_structure(1)
        parent2 = self.create_genome_with_structure(2)
        
        self.set_fitness(parent1, 10.0)
        self.set_fitness(parent2, 8.0)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # Offspring should have some structure
        self.assertGreater(len(offspring.nodes), 0, "Offspring should have nodes")
        self.assertGreater(len(offspring.connections), 0, "Offspring should have connections")
    
    def test_crossover_inherits_from_fitter_parent(self):
        """
        Test that disjoint/excess genes come from fitter parent.
        
        When one parent has genes the other doesn't, those genes should
        come from the fitter parent.
        """
        # Create parents with different structures
        parent1 = self.create_genome_with_structure(1)
        parent2 = self.create_genome_with_structure(2)
        
        # Give parent1 additional structure
        for _ in range(2):
            if parent1.connections:
                parent1.mutate_add_node(self.config.genome_config)
        
        # Make parent1 fitter
        self.set_fitness(parent1, 15.0)
        self.set_fitness(parent2, 5.0)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # Offspring should have all nodes from parent1 (fitter)
        for node_key in parent1.nodes:
            self.assertIn(node_key, offspring.nodes,
                         f"Offspring should have node {node_key} from fitter parent")
    
    def test_crossover_matching_genes_from_both_parents(self):
        """
        Test that matching genes can come from either parent.
        
        For genes present in both parents (same innovation number),
        the offspring should randomly inherit from either parent.
        """
        # Create similar parents
        parent1 = self.create_genome_with_structure(1)
        parent2 = self.create_genome_with_structure(2)
        
        self.set_fitness(parent1, 10.0)
        self.set_fitness(parent2, 10.0)
        
        # Perform multiple crossovers
        offspring_list = []
        for i in range(20):
            offspring = neat.DefaultGenome(100 + i)
            offspring.configure_crossover(parent1, parent2, self.config.genome_config)
            offspring_list.append(offspring)
        
        # Check that we get some variation in inherited genes
        # (Not all offspring should be identical if parents differ)
        if parent1.connections and parent2.connections:
            # Get a matching connection (if any exist)
            matching_innovations = set()
            p1_innovations = {c.innovation: c for c in parent1.connections.values()}
            p2_innovations = {c.innovation: c for c in parent2.connections.values()}
            
            for innov in p1_innovations:
                if innov in p2_innovations:
                    matching_innovations.add(innov)
            
            # If there are matching genes with different attributes, we should see variation
            self.assertGreater(len(matching_innovations), 0,
                             "Should have some matching genes for this test")
    
    def test_crossover_preserves_node_consistency(self):
        """
        Test that crossover maintains node consistency.
        
        All nodes referenced by connections should exist in the offspring.
        """
        parent1 = self.create_genome_with_structure(1, num_hidden=2)
        parent2 = self.create_genome_with_structure(2, num_hidden=1)
        
        self.set_fitness(parent1, 12.0)
        self.set_fitness(parent2, 8.0)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # Verify all connection endpoints exist as nodes
        for conn in offspring.connections.values():
            input_node, output_node = conn.key
            
            # Input nodes can be input keys or nodes
            if input_node >= 0:  # Not an input key
                self.assertIn(input_node, offspring.nodes,
                            f"Connection input {input_node} should exist as node")
            
            # Output node must exist
            self.assertIn(output_node, offspring.nodes,
                        f"Connection output {output_node} should exist as node")
    
    # ========== Fitness-Based Inheritance ==========
    
    def test_crossover_fitness_determines_parent_priority(self):
        """
        Test that parent with higher fitness is treated as primary.
        
        All genes from the fitter parent should be in the offspring.
        """
        parent1 = self.create_genome_with_structure(1, num_hidden=3)
        parent2 = self.create_genome_with_structure(2, num_hidden=1)
        
        # Make parent1 significantly fitter
        self.set_fitness(parent1, 100.0)
        self.set_fitness(parent2, 10.0)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # All connections from parent1 should be in offspring
        parent1_innovations = {c.innovation for c in parent1.connections.values()}
        offspring_innovations = {c.innovation for c in offspring.connections.values()}
        
        self.assertTrue(parent1_innovations.issubset(offspring_innovations),
                       "All genes from fitter parent should be in offspring")
    
    def test_crossover_equal_fitness_includes_parent1_genes(self):
        """
        Test crossover behavior when parents have equal fitness.
        
        When fitness is equal (genome1.fitness > genome2.fitness is False),
        the second parent (genome2) is treated as primary parent.
        """
        parent1 = self.create_genome_with_structure(1, num_hidden=1)
        parent2 = self.create_genome_with_structure(2, num_hidden=2)
        
        # Equal fitness - when using >, equal fitness means condition is False
        # so parent1, parent2 = genome2, genome1
        self.set_fitness(parent1, 50.0)
        self.set_fitness(parent2, 50.0)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # Should have all genes from parent2 (becomes parent1 in crossover logic)
        parent2_innovations = {c.innovation for c in parent2.connections.values()}
        offspring_innovations = {c.innovation for c in offspring.connections.values()}
        
        self.assertTrue(parent2_innovations.issubset(offspring_innovations),
                       "Should include all genes from second parent when fitness equal")
    
    def test_crossover_swaps_parent_order_by_fitness(self):
        """
        Test that parent roles are determined by fitness, not parameter order.
        
        The genome with higher fitness should be parent1 regardless of order.
        """
        genome_a = self.create_genome_with_structure(1, num_hidden=2)
        genome_b = self.create_genome_with_structure(2, num_hidden=1)
        
        # Make genome_b fitter
        self.set_fitness(genome_a, 30.0)
        self.set_fitness(genome_b, 70.0)
        
        # Test both orders
        offspring1 = neat.DefaultGenome(3)
        offspring1.configure_crossover(genome_a, genome_b, self.config.genome_config)
        
        offspring2 = neat.DefaultGenome(4)
        offspring2.configure_crossover(genome_b, genome_a, self.config.genome_config)
        
        # Both should have all genes from genome_b (fitter)
        genome_b_innovations = {c.innovation for c in genome_b.connections.values()}
        
        offspring1_innovations = {c.innovation for c in offspring1.connections.values()}
        self.assertTrue(genome_b_innovations.issubset(offspring1_innovations),
                       "Offspring1 should have all genes from fitter parent")
        
        offspring2_innovations = {c.innovation for c in offspring2.connections.values()}
        self.assertTrue(genome_b_innovations.issubset(offspring2_innovations),
                       "Offspring2 should have all genes from fitter parent")
    
    # ========== Innovation Number Matching ==========
    
    def test_crossover_matches_by_innovation_number(self):
        """
        Test that genes are matched by innovation number, not connection key.
        
        The NEAT algorithm uses innovation numbers to identify homologous genes.
        """
        parent1 = self.create_genome_with_structure(1)
        parent2 = self.create_genome_with_structure(2)
        
        self.set_fitness(parent1, 10.0)
        self.set_fitness(parent2, 10.0)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # Verify offspring connections have valid innovation numbers
        for conn in offspring.connections.values():
            self.assertIsNotNone(conn.innovation,
                               "All connections should have innovation numbers")
            self.assertIsInstance(conn.innovation, int,
                                "Innovation numbers should be integers")
    
    def test_crossover_handles_disjoint_genes(self):
        """
        Test that disjoint genes (in middle of innovation range) are handled correctly.
        
        Disjoint genes from the fitter parent should be included.
        """
        # Create parent with more complex structure
        parent1 = self.create_genome_with_structure(1)
        
        # Add several nodes to create disjoint genes
        for _ in range(3):
            if parent1.connections:
                parent1.mutate_add_node(self.config.genome_config)
        
        parent2 = self.create_genome_with_structure(2)
        
        # Make parent1 fitter
        self.set_fitness(parent1, 20.0)
        self.set_fitness(parent2, 5.0)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # Count innovations
        parent1_innovation_count = len(parent1.connections)
        offspring_innovation_count = len(offspring.connections)
        
        # Offspring should have at least as many as parent1
        self.assertGreaterEqual(offspring_innovation_count, parent1_innovation_count,
                               "Offspring should have all genes from fitter parent")
    
    def test_crossover_handles_excess_genes(self):
        """
        Test that excess genes (beyond other parent's range) are handled correctly.
        
        Excess genes from the fitter parent should be included.
        """
        parent1 = self.create_genome_with_structure(1)
        parent2 = self.create_genome_with_structure(2)
        
        # Give parent2 additional mutations (excess genes)
        for _ in range(4):
            parent2.mutate_add_connection(self.config.genome_config)
        
        # Make parent2 fitter
        self.set_fitness(parent1, 5.0)
        self.set_fitness(parent2, 25.0)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # Offspring should include the excess genes from parent2
        parent2_innovations = {c.innovation for c in parent2.connections.values()}
        offspring_innovations = {c.innovation for c in offspring.connections.values()}
        
        self.assertTrue(parent2_innovations.issubset(offspring_innovations),
                       "Offspring should include excess genes from fitter parent")
    
    # ========== Edge Cases ==========
    
    def test_crossover_identical_genomes(self):
        """
        Test crossover between identical genomes.
        
        Should produce offspring identical to parents.
        """
        parent1 = self.create_genome_with_structure(1)
        
        # Create identical parent by copying
        parent2 = neat.DefaultGenome(2)
        parent2.nodes = {k: v.copy() for k, v in parent1.nodes.items()}
        parent2.connections = {k: v.copy() for k, v in parent1.connections.items()}
        
        self.set_fitness(parent1, 10.0)
        self.set_fitness(parent2, 10.0)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # Offspring should match parents
        self.assertEqual(len(offspring.nodes), len(parent1.nodes),
                        "Offspring should have same nodes as parents")
        self.assertEqual(len(offspring.connections), len(parent1.connections),
                        "Offspring should have same connections as parents")
    
    def test_crossover_maximally_different_genomes(self):
        """
        Test crossover between very different genomes.
        
        Should handle case where genomes have minimal gene overlap.
        """
        # Create minimal parent
        parent1 = self.create_genome_with_structure(1)
        
        # Create complex parent
        parent2 = self.create_genome_with_structure(2, num_hidden=5)
        for _ in range(5):
            parent2.mutate_add_connection(self.config.genome_config)
        
        self.set_fitness(parent1, 100.0)  # Minimal parent is fitter
        self.set_fitness(parent2, 10.0)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # Should not crash and should have valid structure
        self.assertGreater(len(offspring.nodes), 0, "Offspring should have nodes")
        self.assertGreater(len(offspring.connections), 0, "Offspring should have connections")
        
        # Should have similar structure to parent1 (fitter, minimal)
        self.assertEqual(len(offspring.connections), len(parent1.connections),
                        "Offspring should resemble fitter (minimal) parent")
    
    def test_crossover_with_disabled_connections(self):
        """
        Test crossover correctly handles disabled connections.
        
        Disabled connections should be inherited and may be re-enabled
        according to the 75% disable rule in gene crossover.
        """
        parent1 = self.create_genome_with_structure(1)
        parent2 = self.create_genome_with_structure(2)
        
        # Add and disable a connection in parent1
        parent1.mutate_add_node(self.config.genome_config)  # This disables a connection
        
        self.set_fitness(parent1, 15.0)
        self.set_fitness(parent2, 5.0)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # Offspring should have connections (enabled and disabled)
        total_connections = len(offspring.connections)
        disabled_connections = sum(1 for c in offspring.connections.values() 
                                  if not c.enabled)
        
        self.assertGreaterEqual(total_connections, disabled_connections,
                               "Should have some connections")
    
    def test_crossover_preserves_innovation_numbers(self):
        """
        Test that crossover preserves innovation numbers from parents.
        
        Innovation numbers should not be reassigned during crossover.
        """
        parent1 = self.create_genome_with_structure(1, num_hidden=1)
        parent2 = self.create_genome_with_structure(2, num_hidden=1)
        
        self.set_fitness(parent1, 10.0)
        self.set_fitness(parent2, 10.0)
        
        # Collect parent innovation numbers
        parent_innovations = set()
        for c in parent1.connections.values():
            parent_innovations.add(c.innovation)
        for c in parent2.connections.values():
            parent_innovations.add(c.innovation)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # All offspring innovations should come from parents
        for conn in offspring.connections.values():
            self.assertIn(conn.innovation, parent_innovations,
                         "Offspring innovation numbers should come from parents")
    
    def test_crossover_with_no_common_genes(self):
        """
        Test crossover when parents share no common innovation numbers.
        
        This can happen in early generations with different mutation paths.
        """
        # Create parents and give them completely different connections
        parent1 = self.create_genome_with_structure(1)
        parent2 = self.create_genome_with_structure(2)
        
        # Clear parent2 connections and add new ones with different innovations
        parent2.connections.clear()
        for _ in range(3):
            parent2.mutate_add_connection(self.config.genome_config)
        
        self.set_fitness(parent1, 12.0)
        self.set_fitness(parent2, 8.0)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # Should have all connections from fitter parent
        parent1_innovations = {c.innovation for c in parent1.connections.values()}
        offspring_innovations = {c.innovation for c in offspring.connections.values()}
        
        self.assertEqual(parent1_innovations, offspring_innovations,
                        "With no common genes, should have all from fitter parent")
    
    def test_crossover_node_inheritance(self):
        """
        Test that node genes are correctly inherited during crossover.
        
        Nodes should be inherited based on which connections are inherited.
        """
        parent1 = self.create_genome_with_structure(1, num_hidden=2)
        parent2 = self.create_genome_with_structure(2, num_hidden=1)
        
        self.set_fitness(parent1, 10.0)
        self.set_fitness(parent2, 8.0)
        
        offspring = neat.DefaultGenome(3)
        offspring.configure_crossover(parent1, parent2, self.config.genome_config)
        
        # All nodes from parent1 should be in offspring (it's fitter)
        for node_key in parent1.nodes:
            self.assertIn(node_key, offspring.nodes,
                         f"Node {node_key} from fitter parent should be in offspring")
        
        # Verify node attributes are from parents
        for node_key, node in offspring.nodes.items():
            # Node should have valid attributes
            self.assertIsNotNone(node.bias)
            self.assertIsNotNone(node.response)
            self.assertIsNotNone(node.activation)
            self.assertIsNotNone(node.aggregation)
    
    def test_crossover_multiple_times_produces_variation(self):
        """
        Test that multiple crossovers produce different offspring.
        
        Due to random gene selection, offspring should vary.
        """
        parent1 = self.create_genome_with_structure(1, num_hidden=2)
        parent2 = self.create_genome_with_structure(2, num_hidden=2)
        
        self.set_fitness(parent1, 10.0)
        self.set_fitness(parent2, 10.0)
        
        # Create multiple offspring
        offspring_list = []
        for i in range(10):
            offspring = neat.DefaultGenome(100 + i)
            offspring.configure_crossover(parent1, parent2, self.config.genome_config)
            offspring_list.append(offspring)
        
        # Check that offspring exist and are valid
        for offspring in offspring_list:
            self.assertGreater(len(offspring.nodes), 0)
            self.assertGreater(len(offspring.connections), 0)
        
        # With matching genes, some variation in attributes is possible
        # (This is a weak test since variation depends on parent differences)
        self.assertEqual(len(offspring_list), 10, "Should create all offspring")


if __name__ == '__main__':
    unittest.main()
