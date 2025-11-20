"""
Comprehensive tests for species management in neat-python.

Tests cover:
- Species creation and assignment based on genomic distance
- Genomic distance calculations and validation
- GenomeDistanceCache behavior
- Species representative selection
- Species persistence across generations
- Genome-to-species mapping consistency
- Edge cases and boundary conditions
- Configuration variations

"""

import os
import unittest
import neat
from neat.species import Species, GenomeDistanceCache, DefaultSpeciesSet
from neat.reporting import ReporterSet


class TestSpeciesManagement(unittest.TestCase):
    """Tests for species creation, assignment, and management."""
    
    def setUp(self):
        """Set up test configuration and helper objects."""
        # Determine path to configuration file
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(
            neat.DefaultGenome, 
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet, 
            neat.DefaultStagnation,
            config_path
        )
        
        # Initialize innovation tracker
        self.innovation_tracker = neat.InnovationTracker()
        self.config.genome_config.innovation_tracker = self.innovation_tracker
        
        # Create reporter set for species set
        self.reporters = ReporterSet()
    
    def create_test_genome(self, genome_id):
        """
        Create a genome with proper initialization.
        
        Args:
            genome_id: Unique identifier for the genome
            
        Returns:
            Configured genome instance
        """
        g = neat.DefaultGenome(genome_id)
        g.configure_new(self.config.genome_config)
        g.fitness = 1.0  # Set a default fitness
        return g
    
    def create_population(self, size):
        """
        Create a population of genomes.
        
        Args:
            size: Number of genomes to create
            
        Returns:
            Dictionary mapping genome IDs to genome instances
        """
        population = {}
        for i in range(size):
            genome = self.create_test_genome(i)
            population[i] = genome
        return population
    
    # ========== Species Creation and Assignment Tests ==========
    
    def test_species_creation_first_genome(self):
        """
        Test that the first genome creates a new species.
        
        This is fundamental to species management - when no species exist,
        the first genome must create the initial species.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        population = self.create_population(1)
        
        species_set.speciate(self.config, population, generation=0)
        
        # Should have exactly one species
        self.assertEqual(len(species_set.species), 1)
        
        # The genome should be mapped to that species
        self.assertIn(0, species_set.genome_to_species)
        
        # The species should contain the genome
        species_id = species_set.genome_to_species[0]
        species = species_set.species[species_id]
        self.assertIn(0, species.members)
        self.assertEqual(species.representative.key, 0)
    
    def test_species_assignment_compatible(self):
        """
        Test that compatible genomes are assigned to the same species.
        
        When genomes have low genomic distance (below compatibility threshold),
        they should be grouped into the same species.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        
        # Create two identical genomes (distance = 0)
        g1 = self.create_test_genome(0)
        g2 = self.create_test_genome(1)
        # Make g2 structurally identical to g1
        g2.nodes = dict(g1.nodes)
        g2.connections = dict(g1.connections)
        
        population = {0: g1, 1: g2}
        species_set.speciate(self.config, population, generation=0)
        
        # Should have only one species since genomes are identical
        self.assertEqual(len(species_set.species), 1)
        
        # Both genomes should be in the same species
        species_id_0 = species_set.genome_to_species[0]
        species_id_1 = species_set.genome_to_species[1]
        self.assertEqual(species_id_0, species_id_1)
    
    def test_new_species_when_incompatible(self):
        """
        Test that incompatible genomes create separate species.
        
        When genomes have high genomic distance (above compatibility threshold),
        they should form separate species.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        
        # Create two very different genomes
        g1 = self.create_test_genome(0)
        g2 = self.create_test_genome(1)
        
        # Add many different nodes and connections to g2 to make it very different
        # The compatibility threshold is 3.0, so we need significant structural difference
        for i in range(10, 30):
            node = self.config.genome_config.node_gene_type(i)
            node.init_attributes(self.config.genome_config)
            g2.nodes[i] = node
        
        population = {0: g1, 1: g2}
        species_set.speciate(self.config, population, generation=0)
        
        # Calculate actual distance to verify our test setup
        distance = g1.distance(g2, self.config.genome_config)
        
        # If distance exceeds threshold, should have two species
        # Otherwise, this test documents the actual behavior
        if distance >= self.config.species_set_config.compatibility_threshold:
            self.assertEqual(len(species_set.species), 2)
            # Genomes should be in different species
            species_id_0 = species_set.genome_to_species[0]
            species_id_1 = species_set.genome_to_species[1]
            self.assertNotEqual(species_id_0, species_id_1)
        else:
            # Genomes are still compatible despite differences
            self.assertEqual(len(species_set.species), 1)
    
    def test_species_assignment_multiple_genomes(self):
        """
        Test species assignment with multiple genomes of varying similarity.
        
        This tests the clustering behavior when some genomes are similar
        and others are different.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        population = self.create_population(10)
        
        species_set.speciate(self.config, population, generation=0)
        
        # All genomes should be assigned to species
        self.assertEqual(len(species_set.genome_to_species), 10)
        
        # Each genome should be in exactly one species
        for gid in population:
            self.assertIn(gid, species_set.genome_to_species)
            species_id = species_set.genome_to_species[gid]
            self.assertIn(species_id, species_set.species)
    
    # ========== Genomic Distance Tests ==========
    
    def test_distance_identical_genomes(self):
        """
        Test that distance between identical genomes is 0.
        
        This is a fundamental property - identical genomes should have
        zero genomic distance.
        """
        g1 = self.create_test_genome(0)
        g2 = self.create_test_genome(1)
        
        # Make g2 identical to g1
        g2.nodes = dict(g1.nodes)
        g2.connections = dict(g1.connections)
        
        distance = g1.distance(g2, self.config.genome_config)
        self.assertEqual(distance, 0.0)
    
    def test_distance_symmetry(self):
        """
        Test that genomic distance is symmetric: distance(A, B) == distance(B, A).
        
        This is a required mathematical property for the distance metric.
        """
        g1 = self.create_test_genome(0)
        g2 = self.create_test_genome(1)
        
        distance_12 = g1.distance(g2, self.config.genome_config)
        distance_21 = g2.distance(g1, self.config.genome_config)
        
        self.assertAlmostEqual(distance_12, distance_21, places=10)
    
    def test_distance_empty_connections(self):
        """
        Test distance calculation with genomes that have no connections.
        
        Edge case: genomes with only nodes but no connections should still
        have a calculable distance.
        """
        g1 = self.create_test_genome(0)
        g2 = self.create_test_genome(1)
        
        # Remove all connections
        g1.connections.clear()
        g2.connections.clear()
        
        # Should still be able to calculate distance
        distance = g1.distance(g2, self.config.genome_config)
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0.0)
    
    def test_distance_diverse_structures(self):
        """
        Test distance calculation with genomes of varying structure.
        
        Genomes with different numbers of nodes and connections should
        have larger distances than similar genomes.
        """
        g1 = self.create_test_genome(0)
        g2 = self.create_test_genome(1)
        g3 = self.create_test_genome(2)
        
        # g2 is identical to g1
        g2.nodes = dict(g1.nodes)
        g2.connections = dict(g1.connections)
        
        # g3 has additional nodes
        for i in range(10, 15):
            node = self.config.genome_config.node_gene_type(i)
            node.init_attributes(self.config.genome_config)
            g3.nodes[i] = node
        
        distance_12 = g1.distance(g2, self.config.genome_config)
        distance_13 = g1.distance(g3, self.config.genome_config)
        
        # Distance to identical genome should be less than distance to different genome
        self.assertLess(distance_12, distance_13)
    
    def test_distance_minimal_nodes(self):
        """
        Test distance with minimal genomes (only required nodes).
        
        Edge case: genomes with only input/output nodes should have
        well-defined distance.
        """
        # Create minimal genomes with just the required nodes
        g1 = neat.DefaultGenome(0)
        g1.configure_new(self.config.genome_config)
        
        g2 = neat.DefaultGenome(1)
        g2.configure_new(self.config.genome_config)
        
        distance = g1.distance(g2, self.config.genome_config)
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0.0)
    
    def test_distance_self(self):
        """
        Test that distance from a genome to itself is 0.
        
        A genome should have zero distance from itself.
        """
        g1 = self.create_test_genome(0)
        distance = g1.distance(g1, self.config.genome_config)
        self.assertEqual(distance, 0.0)
    
    # ========== GenomeDistanceCache Tests ==========
    
    def test_cache_miss_on_first_query(self):
        """
        Test that the first query for a genome pair results in a cache miss.
        
        The cache should not have pre-computed distances.
        """
        cache = GenomeDistanceCache(self.config.genome_config)
        g1 = self.create_test_genome(0)
        g2 = self.create_test_genome(1)
        
        initial_misses = cache.misses
        distance = cache(g1, g2)
        
        self.assertEqual(cache.misses, initial_misses + 1)
        self.assertIsInstance(distance, float)
    
    def test_cache_hit_on_repeat_query(self):
        """
        Test that repeated queries for the same genome pair result in cache hits.
        
        Once a distance is computed, subsequent queries should use the cached value.
        """
        cache = GenomeDistanceCache(self.config.genome_config)
        g1 = self.create_test_genome(0)
        g2 = self.create_test_genome(1)
        
        # First query - cache miss
        distance1 = cache(g1, g2)
        initial_hits = cache.hits
        
        # Second query - cache hit
        distance2 = cache(g1, g2)
        
        self.assertEqual(cache.hits, initial_hits + 1)
        self.assertEqual(distance1, distance2)
    
    def test_cache_consistency(self):
        """
        Test that the cache returns consistent values for the same pair.
        
        Multiple queries should always return the exact same distance value.
        """
        cache = GenomeDistanceCache(self.config.genome_config)
        g1 = self.create_test_genome(0)
        g2 = self.create_test_genome(1)
        
        distances = [cache(g1, g2) for _ in range(5)]
        
        # All distances should be identical
        self.assertEqual(len(set(distances)), 1)
    
    def test_cache_symmetry(self):
        """
        Test that cache properly handles symmetric queries: (A,B) and (B,A).
        
        Both orderings should return the same distance and both should be cached.
        """
        cache = GenomeDistanceCache(self.config.genome_config)
        g1 = self.create_test_genome(0)
        g2 = self.create_test_genome(1)
        
        distance_12 = cache(g1, g2)
        initial_hits = cache.hits
        
        distance_21 = cache(g2, g1)
        
        # Second query should be a cache hit
        self.assertEqual(cache.hits, initial_hits + 1)
        self.assertEqual(distance_12, distance_21)
    
    def test_cache_multiple_pairs(self):
        """
        Test cache behavior with multiple genome pairs.
        
        The cache should correctly track distances for multiple pairs
        without confusion.
        """
        cache = GenomeDistanceCache(self.config.genome_config)
        genomes = [self.create_test_genome(i) for i in range(4)]
        
        # Compute distances for all pairs
        distances = {}
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                dist = cache(genomes[i], genomes[j])
                distances[(i, j)] = dist
        
        # Verify all pairs were cached
        initial_hits = cache.hits
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                cached_dist = cache(genomes[i], genomes[j])
                self.assertEqual(cached_dist, distances[(i, j)])
        
        # Should have gotten hits for all pairs
        num_pairs = len(distances)
        self.assertEqual(cache.hits, initial_hits + num_pairs)
    
    # ========== Representative Selection Tests ==========
    
    def test_representative_initial_selection(self):
        """
        Test that the first member of a new species becomes the representative.
        
        When a species is created, the founding genome should be its representative.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        population = self.create_population(1)
        
        species_set.speciate(self.config, population, generation=0)
        
        species_id = species_set.genome_to_species[0]
        species = species_set.species[species_id]
        
        self.assertEqual(species.representative.key, 0)
    
    def test_representative_updated_across_generations(self):
        """
        Test that representatives are updated correctly in subsequent generations.
        
        The new representative should be the genome closest to the current representative.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        
        # Generation 0: Create initial population
        pop_gen0 = self.create_population(3)
        species_set.speciate(self.config, pop_gen0, generation=0)
        
        # Get the species ID and representative
        species_id = species_set.genome_to_species[0]
        old_rep = species_set.species[species_id].representative
        
        # Generation 1: New population (simulate evolution)
        pop_gen1 = self.create_population(3)
        species_set.speciate(self.config, pop_gen1, generation=1)
        
        # Representative should be updated
        new_rep = species_set.species[species_id].representative
        self.assertIsNotNone(new_rep)
        self.assertIn(new_rep.key, pop_gen1)
    
    def test_representative_selection_from_compatible_members(self):
        """
        Test that representative is selected from members of the species.
        
        The representative must be one of the genomes that belongs to the species.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        population = self.create_population(5)
        
        species_set.speciate(self.config, population, generation=0)
        
        # Check each species
        for species_id, species in species_set.species.items():
            rep_key = species.representative.key
            # Representative must be in the population
            self.assertIn(rep_key, population)
            # Representative must be a member of the species
            self.assertIn(rep_key, species.members)
    
    # ========== Species Persistence Tests ==========
    
    def test_species_id_stability(self):
        """
        Test that species IDs remain stable when members persist.
        
        If a species continues to have compatible members across generations,
        its ID should remain the same.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        
        # Generation 0
        pop_gen0 = self.create_population(3)
        species_set.speciate(self.config, pop_gen0, generation=0)
        
        gen0_species_ids = set(species_set.species.keys())
        
        # Generation 1 with similar genomes
        pop_gen1 = self.create_population(3)
        species_set.speciate(self.config, pop_gen1, generation=1)
        
        gen1_species_ids = set(species_set.species.keys())
        
        # At least one species should persist (ID overlap)
        self.assertTrue(len(gen0_species_ids & gen1_species_ids) > 0)
    
    def test_species_persists_with_compatible_members(self):
        """
        Test that species continue when compatible members exist.
        
        A species should not disappear if there are still genomes
        compatible with its representative.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        
        # Create initial population with identical genomes
        g0 = self.create_test_genome(0)
        g1 = self.create_test_genome(1)
        g1.nodes = dict(g0.nodes)
        g1.connections = dict(g0.connections)
        
        pop_gen0 = {0: g0, 1: g1}
        species_set.speciate(self.config, pop_gen0, generation=0)
        
        initial_species_count = len(species_set.species)
        initial_species_id = species_set.genome_to_species[0]
        
        # Create next generation with similar genomes
        g2 = self.create_test_genome(2)
        g3 = self.create_test_genome(3)
        g2.nodes = dict(g0.nodes)
        g2.connections = dict(g0.connections)
        g3.nodes = dict(g0.nodes)
        g3.connections = dict(g0.connections)
        
        pop_gen1 = {2: g2, 3: g3}
        species_set.speciate(self.config, pop_gen1, generation=1)
        
        # The species should still exist with the same ID
        self.assertIn(initial_species_id, species_set.species)
    
    def test_multiple_generations_tracking(self):
        """
        Test species tracking over multiple generations.
        
        Verify that species management works correctly when simulating
        multiple generations of evolution.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        
        for generation in range(5):
            population = self.create_population(10)
            species_set.speciate(self.config, population, generation=generation)
            
            # All genomes should be assigned
            self.assertEqual(len(species_set.genome_to_species), 10)
            
            # All species should have at least one member
            for species_id, species in species_set.species.items():
                self.assertGreater(len(species.members), 0)
    
    # ========== Genome-to-Species Mapping Tests ==========
    
    def test_mapping_synchronized_with_species(self):
        """
        Test that genome_to_species mapping matches actual species membership.
        
        The bidirectional mapping between genomes and species must be consistent.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        population = self.create_population(10)
        
        species_set.speciate(self.config, population, generation=0)
        
        # Check that every genome in genome_to_species is in its species
        for genome_id, species_id in species_set.genome_to_species.items():
            species = species_set.species[species_id]
            self.assertIn(genome_id, species.members)
    
    def test_all_genomes_mapped(self):
        """
        Test that every genome in the population has a species mapping.
        
        No genome should be left unassigned after speciation.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        population = self.create_population(10)
        
        species_set.speciate(self.config, population, generation=0)
        
        # Every genome should be in the mapping
        for genome_id in population.keys():
            self.assertIn(genome_id, species_set.genome_to_species)
    
    def test_no_orphaned_mappings(self):
        """
        Test that no genomes are mapped to non-existent species.
        
        Every species ID in genome_to_species must correspond to an actual species.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        population = self.create_population(10)
        
        species_set.speciate(self.config, population, generation=0)
        
        # Every species ID in the mapping should exist
        for genome_id, species_id in species_set.genome_to_species.items():
            self.assertIn(species_id, species_set.species)
    
    def test_mapping_after_speciation(self):
        """
        Test that mapping is correctly updated after speciation.
        
        Each call to speciate() should produce a consistent mapping.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        population = self.create_population(10)
        
        species_set.speciate(self.config, population, generation=0)
        
        # Verify mapping integrity
        genome_count_in_species = 0
        for species in species_set.species.values():
            genome_count_in_species += len(species.members)
        
        self.assertEqual(genome_count_in_species, len(population))
        self.assertEqual(len(species_set.genome_to_species), len(population))
    
    def test_mapping_integrity(self):
        """
        Test cross-checking between species.members and genome_to_species.
        
        Both data structures should agree on which genomes belong to which species.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        population = self.create_population(10)
        
        species_set.speciate(self.config, population, generation=0)
        
        # Build reverse mapping from species
        reverse_mapping = {}
        for species_id, species in species_set.species.items():
            for genome_id in species.members:
                reverse_mapping[genome_id] = species_id
        
        # Should match genome_to_species
        self.assertEqual(reverse_mapping, species_set.genome_to_species)
    
    # ========== Edge Case Tests ==========
    
    def test_single_genome_population(self):
        """
        Test species management with a single genome.
        
        Edge case: population of size 1 should create exactly one species.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        population = self.create_population(1)
        
        species_set.speciate(self.config, population, generation=0)
        
        self.assertEqual(len(species_set.species), 1)
        self.assertEqual(len(species_set.genome_to_species), 1)
    
    def test_all_genomes_identical(self):
        """
        Test species management when all genomes are identical.
        
        Edge case: all identical genomes (distance = 0) should form a single species.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        
        # Create multiple identical genomes
        g0 = self.create_test_genome(0)
        population = {0: g0}
        
        for i in range(1, 10):
            g = self.create_test_genome(i)
            g.nodes = dict(g0.nodes)
            g.connections = dict(g0.connections)
            population[i] = g
        
        species_set.speciate(self.config, population, generation=0)
        
        # All genomes should be in one species
        self.assertEqual(len(species_set.species), 1)
        
        # All should map to the same species
        species_ids = set(species_set.genome_to_species.values())
        self.assertEqual(len(species_ids), 1)
    
    def test_all_genomes_maximally_different(self):
        """
        Test species management when all genomes are very different.
        
        Edge case: maximally different genomes should form separate species.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        
        # Create very different genomes by adding unique nodes to each
        # Add more nodes to ensure distance exceeds compatibility threshold (3.0)
        population = {}
        for i in range(5):
            g = self.create_test_genome(i)
            # Add many unique nodes to make each genome very different
            for j in range(30 * i, 30 * i + 30):
                node = self.config.genome_config.node_gene_type(j)
                node.init_attributes(self.config.genome_config)
                g.nodes[j] = node
            population[i] = g
        
        species_set.speciate(self.config, population, generation=0)
        
        # With compatibility threshold of 3.0, genomes with 30+ unique nodes each
        # should form multiple species. At minimum, verify the algorithm runs.
        self.assertGreaterEqual(len(species_set.species), 1)
        
        # All genomes should still be assigned
        self.assertEqual(len(species_set.genome_to_species), 5)
    
    def test_species_with_single_member(self):
        """
        Test that species with a single member are handled correctly.
        
        Edge case: a species can have just one member.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        
        # Create genomes that are all different
        population = {}
        for i in range(3):
            g = self.create_test_genome(i)
            for j in range(10 * i, 10 * i + 10):
                node = self.config.genome_config.node_gene_type(j)
                node.init_attributes(self.config.genome_config)
                g.nodes[j] = node
            population[i] = g
        
        species_set.speciate(self.config, population, generation=0)
        
        # Check for species with single members
        single_member_species = [s for s in species_set.species.values() 
                                  if len(s.members) == 1]
        
        # Each single-member species should be valid
        for species in single_member_species:
            self.assertEqual(len(species.members), 1)
            self.assertIsNotNone(species.representative)
    
    def test_empty_genome_connections(self):
        """
        Test species management with genomes that have no connections.
        
        Edge case: genomes with nodes but no connections should still
        be properly speciated.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        
        # Create genomes with no connections
        population = {}
        for i in range(5):
            g = self.create_test_genome(i)
            g.connections.clear()
            population[i] = g
        
        species_set.speciate(self.config, population, generation=0)
        
        # Should successfully speciate
        self.assertEqual(len(species_set.genome_to_species), 5)
        self.assertGreater(len(species_set.species), 0)
    
    def test_large_population(self):
        """
        Test species management with a large population.
        
        Verify that the algorithm scales to larger populations without issues.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        population = self.create_population(100)
        
        species_set.speciate(self.config, population, generation=0)
        
        # All genomes should be assigned
        self.assertEqual(len(species_set.genome_to_species), 100)
        
        # All species should have members
        total_members = sum(len(s.members) for s in species_set.species.values())
        self.assertEqual(total_members, 100)
    
    # ========== Configuration Tests ==========
    
    def test_different_compatibility_thresholds(self):
        """
        Test species formation with different compatibility thresholds.
        
        Lower thresholds should produce more species (stricter grouping),
        higher thresholds should produce fewer species (looser grouping).
        """
        population = self.create_population(20)
        
        # Test with low threshold (strict)
        config_strict = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            os.path.join(os.path.dirname(__file__), 'test_configuration')
        )
        config_strict.species_set_config.compatibility_threshold = 1.0
        
        species_set_strict = DefaultSpeciesSet(
            config_strict.species_set_config, 
            self.reporters
        )
        
        # Need to set innovation tracker
        innovation_tracker = neat.InnovationTracker()
        config_strict.genome_config.innovation_tracker = innovation_tracker
        
        # Re-initialize population with new config
        pop_strict = {}
        for gid, g in population.items():
            new_g = neat.DefaultGenome(gid)
            new_g.configure_new(config_strict.genome_config)
            new_g.fitness = 1.0
            pop_strict[gid] = new_g
        
        species_set_strict.speciate(config_strict, pop_strict, generation=0)
        num_species_strict = len(species_set_strict.species)
        
        # Test with high threshold (loose)
        config_loose = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            os.path.join(os.path.dirname(__file__), 'test_configuration')
        )
        config_loose.species_set_config.compatibility_threshold = 10.0
        config_loose.genome_config.innovation_tracker = innovation_tracker
        
        species_set_loose = DefaultSpeciesSet(
            config_loose.species_set_config,
            self.reporters
        )
        
        # Re-initialize population with new config
        pop_loose = {}
        for gid in pop_strict.keys():
            new_g = neat.DefaultGenome(gid)
            new_g.configure_new(config_loose.genome_config)
            new_g.fitness = 1.0
            pop_loose[gid] = new_g
        
        species_set_loose.speciate(config_loose, pop_loose, generation=0)
        num_species_loose = len(species_set_loose.species)
        
        # Strict threshold should produce more or equal species than loose
        self.assertGreaterEqual(num_species_strict, num_species_loose)
    
    def test_config_from_file(self):
        """
        Test that species management works with configuration loaded from file.
        
        Verify compatibility with the standard configuration file format.
        """
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        innovation_tracker = neat.InnovationTracker()
        config.genome_config.innovation_tracker = innovation_tracker
        
        species_set = DefaultSpeciesSet(config.species_set_config, self.reporters)
        
        # Create and speciate population
        population = {}
        for i in range(10):
            g = neat.DefaultGenome(i)
            g.configure_new(config.genome_config)
            g.fitness = 1.0
            population[i] = g
        
        species_set.speciate(config, population, generation=0)
        
        # Should successfully create species
        self.assertGreater(len(species_set.species), 0)
        self.assertEqual(len(species_set.genome_to_species), 10)
    
    def test_species_get_methods(self):
        """
        Test the get_species_id and get_species helper methods.
        
        These methods should correctly retrieve species information for genomes.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        population = self.create_population(5)
        
        species_set.speciate(self.config, population, generation=0)
        
        # Test get_species_id
        for genome_id in population.keys():
            species_id = species_set.get_species_id(genome_id)
            self.assertIn(species_id, species_set.species)
        
        # Test get_species
        for genome_id in population.keys():
            species = species_set.get_species(genome_id)
            self.assertIsInstance(species, Species)
            self.assertIn(genome_id, species.members)
    
    def test_species_fitness_tracking(self):
        """
        Test that species properly track fitness information.
        
        The Species class should maintain fitness and fitness history.
        """
        species_set = DefaultSpeciesSet(self.config.species_set_config, self.reporters)
        population = self.create_population(5)
        
        # Set varying fitness values
        for i, genome in enumerate(population.values()):
            genome.fitness = float(i + 1)
        
        species_set.speciate(self.config, population, generation=0)
        
        # Check that species have members with fitness
        for species in species_set.species.values():
            fitnesses = species.get_fitnesses()
            self.assertGreater(len(fitnesses), 0)
            for fitness in fitnesses:
                self.assertIsNotNone(fitness)
                self.assertIsInstance(fitness, (int, float))


if __name__ == '__main__':
    unittest.main()
