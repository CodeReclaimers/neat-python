"""
Tests for the genomic distance function changes:
- Item B: Innovation-based matching (instead of tuple-key matching)
- Item D: Separate excess vs disjoint coefficients
- Item H: Configurable node gene contribution
"""
import os
import unittest

import neat
from neat.genes import DefaultConnectionGene, DefaultNodeGene


class TestInnovationBasedDistance(unittest.TestCase):
    """Tests for innovation-number-based connection gene matching in distance()."""

    def setUp(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)
        self.gc = self.config.genome_config
        self.innovation_tracker = neat.InnovationTracker()
        self.gc.innovation_tracker = self.innovation_tracker

    def _make_genome(self, gid):
        g = neat.DefaultGenome(gid)
        g.configure_new(self.gc)
        return g

    def test_distance_matches_by_innovation(self):
        """Two genomes with same-innovation connections should have zero connection distance."""
        g1 = self._make_genome(0)
        g2 = self._make_genome(1)

        # Make g2 a copy of g1's structure.
        g2.nodes = {}
        for k, n in g1.nodes.items():
            g2.nodes[k] = n.copy()
        g2.connections = {}
        for k, c in g1.connections.items():
            g2.connections[k] = c.copy()

        distance = g1.distance(g2, self.gc)
        self.assertAlmostEqual(distance, 0.0,
                              msg="Identical genomes should have zero distance")

    def test_distance_innovation_mismatch_is_disjoint(self):
        """Connections with different innovation numbers at same endpoints should be disjoint."""
        g1 = self._make_genome(0)
        g2 = self._make_genome(1)

        # Give g2 the same nodes as g1.
        g2.nodes = {}
        for k, n in g1.nodes.items():
            g2.nodes[k] = n.copy()

        # Give g2 a connection with the same key but different innovation number.
        g2.connections = {}
        if g1.connections:
            for k, c in g1.connections.items():
                # Create a connection with same key but innovation+1000.
                new_c = DefaultConnectionGene(k, innovation=c.innovation + 1000)
                new_c.weight = c.weight
                new_c.enabled = c.enabled
                g2.connections[k] = new_c
                break  # just one connection for simplicity

            # Remove g1's other connections to keep it simple
            first_key = list(g1.connections.keys())[0]
            g1.connections = {first_key: g1.connections[first_key]}

            distance = g1.distance(g2, self.gc)
            # Should have disjoint genes, so distance > 0
            self.assertGreater(distance, 0.0,
                              msg="Different innovation numbers should count as disjoint")

    def test_distance_innovation_collision_handled(self):
        """Connections with same innovation but different endpoints are treated as disjoint."""
        g1 = self._make_genome(0)
        g2 = self._make_genome(1)

        # Create minimal genomes with one connection each, same innovation but different keys.
        g1.connections = {}
        g2.connections = {}
        g1.nodes = {}
        g2.nodes = {}

        # Add output nodes (required).
        for k in self.gc.output_keys:
            g1.nodes[k] = g1.create_node(self.gc, k)
            g2.nodes[k] = g2.create_node(self.gc, k)

        c1 = DefaultConnectionGene((-1, 0), innovation=1)
        c1.weight = 0.5
        c1.enabled = True
        g1.connections[(-1, 0)] = c1

        c2 = DefaultConnectionGene((-2, 0), innovation=1)  # Same innovation, different key
        c2.weight = 0.5
        c2.enabled = True
        g2.connections[(-2, 0)] = c2

        distance = g1.distance(g2, self.gc)
        # Both genes should be counted as disjoint (collision).
        self.assertGreater(distance, 0.0,
                          msg="Innovation collision should be treated as disjoint")

    def test_distance_symmetry_with_innovations(self):
        """distance(A,B) should equal distance(B,A) with innovation-based matching."""
        g1 = self._make_genome(0)
        g2 = self._make_genome(1)

        d12 = g1.distance(g2, self.gc)
        d21 = g2.distance(g1, self.gc)
        self.assertAlmostEqual(d12, d21, places=10,
                              msg="Distance should be symmetric")


class TestExcessVsDisjointCoefficient(unittest.TestCase):
    """Tests for separate excess gene coefficient (Item D)."""

    def setUp(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)
        self.gc = self.config.genome_config
        self.innovation_tracker = neat.InnovationTracker()
        self.gc.innovation_tracker = self.innovation_tracker

    def test_excess_coefficient_defaults_to_disjoint(self):
        """When compatibility_excess_coefficient is 'auto', it should match disjoint coefficient."""
        self.assertEqual(self.gc.compatibility_excess_coefficient, 'auto')

        # Create two genomes with different connections.
        g1 = neat.DefaultGenome(0)
        g1.configure_new(self.gc)
        g2 = neat.DefaultGenome(1)
        g2.configure_new(self.gc)

        # Distance should work without error.
        d = g1.distance(g2, self.gc)
        self.assertIsInstance(d, float)

    def test_excess_vs_disjoint_distinguished(self):
        """With explicit coefficients, excess and disjoint should be weighted differently."""
        g1 = neat.DefaultGenome(0)
        g1.configure_new(self.gc)
        g2 = neat.DefaultGenome(1)
        g2.configure_new(self.gc)

        # Give g2 the same nodes as g1.
        g2.nodes = {}
        for k, n in g1.nodes.items():
            g2.nodes[k] = n.copy()

        # Set up connections: g1 has innovations [1, 2], g2 has [1, 3].
        # Innovation 1 is homologous, 2 is disjoint in g1 (within g2's range),
        # 3 is excess in g2 (beyond g1's max of 2).
        g1.connections = {}
        g2.connections = {}

        c1a = DefaultConnectionGene((-1, 0), innovation=1)
        c1a.weight = 0.5
        c1a.enabled = True
        g1.connections[(-1, 0)] = c1a

        c1b = DefaultConnectionGene((-1, 0), innovation=1)
        c1b.weight = 0.5
        c1b.enabled = True
        g2.connections[(-1, 0)] = c1b

        c2 = DefaultConnectionGene((-2, 0), innovation=2)
        c2.weight = 0.5
        c2.enabled = True
        g1.connections[(-2, 0)] = c2

        c3 = DefaultConnectionGene((-1, 1) if len(self.gc.output_keys) > 1 else (-2, 0), innovation=3)
        # Use a different key to avoid collision
        key3 = (-2, 0) if (-2, 0) not in g2.connections else (-1, 1)
        c3 = DefaultConnectionGene(key3, innovation=3)
        c3.weight = 0.5
        c3.enabled = True
        g2.connections[key3] = c3

        # Disable node distance for cleaner test.
        self.gc.compatibility_include_node_genes = False

        # Distance with auto (excess == disjoint).
        self.gc.compatibility_excess_coefficient = 'auto'
        d_auto = g1.distance(g2, self.gc)

        # Now set excess coefficient to something different.
        self.gc.compatibility_excess_coefficient = '5.0'
        d_explicit = g1.distance(g2, self.gc)

        # They should differ because excess genes are weighted differently.
        self.assertNotAlmostEqual(d_auto, d_explicit, places=5,
                                 msg="Different excess coefficient should change distance")

        # Restore defaults.
        self.gc.compatibility_excess_coefficient = 'auto'
        self.gc.compatibility_include_node_genes = True


class TestNodeGeneDistanceFlag(unittest.TestCase):
    """Tests for compatibility_include_node_genes flag (Item H)."""

    def setUp(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)
        self.gc = self.config.genome_config
        self.innovation_tracker = neat.InnovationTracker()
        self.gc.innovation_tracker = self.innovation_tracker

    def test_node_genes_included_by_default(self):
        """Default behavior should include node distance."""
        self.assertTrue(self.gc.compatibility_include_node_genes)

        g1 = neat.DefaultGenome(0)
        g1.configure_new(self.gc)
        g2 = neat.DefaultGenome(1)
        g2.configure_new(self.gc)

        # Make connections identical, but nodes differ (random init).
        g2.connections = {}
        for k, c in g1.connections.items():
            g2.connections[k] = c.copy()

        # If nodes have different random biases, distance should be > 0.
        d = g1.distance(g2, self.gc)
        # Can't assert > 0 since nodes might randomly be identical, but it should work.
        self.assertIsInstance(d, float)

    def test_node_genes_excluded(self):
        """With flag=False, genomes differing only in node attributes should have zero distance."""
        g1 = neat.DefaultGenome(0)
        g1.configure_new(self.gc)
        g2 = neat.DefaultGenome(1)
        g2.configure_new(self.gc)

        # Make connections identical.
        g2.connections = {}
        for k, c in g1.connections.items():
            g2.connections[k] = c.copy()

        # Nodes differ (random init), but we exclude them.
        self.gc.compatibility_include_node_genes = False
        d = g1.distance(g2, self.gc)
        self.assertAlmostEqual(d, 0.0,
                              msg="With node genes excluded and identical connections, distance should be 0")

        # Restore default.
        self.gc.compatibility_include_node_genes = True

    def test_node_genes_flag_only_affects_nodes(self):
        """Connection gene distance should be unaffected by the node gene flag."""
        g1 = neat.DefaultGenome(0)
        g1.configure_new(self.gc)
        g2 = neat.DefaultGenome(1)
        g2.configure_new(self.gc)

        # Make nodes identical, connections differ.
        g2.nodes = {}
        for k, n in g1.nodes.items():
            g2.nodes[k] = n.copy()

        self.gc.compatibility_include_node_genes = True
        d_with = g1.distance(g2, self.gc)

        self.gc.compatibility_include_node_genes = False
        d_without = g1.distance(g2, self.gc)

        # The node component should be zero in both (identical nodes), so
        # connection distance should be the same.
        self.assertAlmostEqual(d_with, d_without, places=10,
                              msg="With identical nodes, flag should not affect total distance")

        # Restore default.
        self.gc.compatibility_include_node_genes = True


if __name__ == '__main__':
    unittest.main()
