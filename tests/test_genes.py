import unittest

from neat.genes import DefaultNodeGene, DefaultConnectionGene


class _DummyConfig:
    """Minimal config object for distance calculations."""
    compatibility_weight_coefficient = 0.5


class TestDefaultNodeGene(unittest.TestCase):
    def test_init_requires_int_key(self):
        """DefaultNodeGene should enforce integer keys via assertion."""
        with self.assertRaises(AssertionError):
            DefaultNodeGene("not-an-int")

    def test_lt_compares_by_key_and_rejects_mismatched_key_types(self):
        """Genes order by their key type and value; mixed key types are rejected."""
        a = DefaultNodeGene(-1)
        b = DefaultNodeGene(0)

        # Same key type: comparison uses the underlying key.
        self.assertTrue(a < b)

        # Mismatched key types should trigger the defensive assertion.
        conn = DefaultConnectionGene((0, 1), innovation=1)
        with self.assertRaises(AssertionError):
            _ = a < conn

    def test_str_includes_key_and_attributes(self):
        """__str__ should include class name, key, and attribute values."""
        g = DefaultNodeGene(0)
        g.bias = 0.5
        g.response = 1.0
        g.activation = "relu"
        g.aggregation = "sum"

        s = str(g)
        self.assertIn("DefaultNodeGene", s)
        self.assertIn("key=0", s)
        self.assertIn("bias=0.5", s)
        self.assertIn("response=1.0", s)
        self.assertIn("activation=relu", s)
        self.assertIn("aggregation=sum", s)

    def test_distance_respects_activation_aggregation_and_scaling(self):
        """Node distance should include bias/response plus activation/aggregation mismatches."""
        cfg = _DummyConfig()

        g1 = DefaultNodeGene(0)
        g2 = DefaultNodeGene(0)

        # Same activation/aggregation, different bias/response.
        g1.bias = 0.5
        g1.response = 1.0
        g1.activation = "relu"
        g1.aggregation = "sum"

        g2.bias = -0.5
        g2.response = 0.0
        g2.activation = "relu"
        g2.aggregation = "sum"

        # |0.5 - (-0.5)| + |1.0 - 0.0| = 1.0 + 1.0 = 2.0, scaled by 0.5.
        self.assertEqual(g1.distance(g2, cfg), 1.0)

        # Change activation and aggregation to introduce two extra unit penalties.
        g2.activation = "sigmoid"
        g2.aggregation = "product"

        # Base distance 2.0 + 1 + 1 = 4.0, scaled by 0.5.
        self.assertEqual(g1.distance(g2, cfg), 2.0)

    def test_copy_preserves_key_and_attributes(self):
        """BaseGene.copy should produce a new node gene with identical attributes."""
        g = DefaultNodeGene(0)
        g.bias = -0.25
        g.response = 2.0
        g.activation = "tanh"
        g.aggregation = "sum"

        clone = g.copy()
        self.assertIsNot(clone, g)
        self.assertIsInstance(clone, DefaultNodeGene)
        self.assertEqual(clone.key, g.key)
        self.assertEqual(clone.bias, g.bias)
        self.assertEqual(clone.response, g.response)
        self.assertEqual(clone.activation, g.activation)
        self.assertEqual(clone.aggregation, g.aggregation)

    def test_crossover_randomly_inherits_attributes(self):
        """Node gene crossover should inherit each attribute from one parent or the other."""
        g1 = DefaultNodeGene(0)
        g1.bias = -1.0
        g1.response = 0.5
        g1.activation = "relu"
        g1.aggregation = "sum"

        g2 = DefaultNodeGene(0)
        g2.bias = 1.0
        g2.response = 2.0
        g2.activation = "sigmoid"
        g2.aggregation = "product"

        child = g1.crossover(g2)

        self.assertEqual(child.key, 0)
        self.assertIn(child.bias, {-1.0, 1.0})
        self.assertIn(child.response, {0.5, 2.0})
        self.assertIn(child.activation, {"relu", "sigmoid"})
        self.assertIn(child.aggregation, {"sum", "product"})


class TestDefaultConnectionGene(unittest.TestCase):
    def test_init_requires_tuple_key_and_int_innovation(self):
        """Connection genes require a tuple key and integer innovation number."""
        with self.assertRaises(AssertionError):
            DefaultConnectionGene("not-a-tuple", innovation=1)

        with self.assertRaises(AssertionError):
            DefaultConnectionGene((0, 1), innovation=None)

        with self.assertRaises(AssertionError):
            DefaultConnectionGene((0, 1), innovation="not-int")

    def test_equality_and_hash_based_on_innovation_only(self):
        """Connection genes compare and hash by innovation number, not key or weight."""
        g1 = DefaultConnectionGene((0, 1), innovation=1)
        g2 = DefaultConnectionGene((2, 3), innovation=1)
        g3 = DefaultConnectionGene((0, 1), innovation=2)

        g1.weight = 0.1
        g2.weight = 0.9
        g3.weight = 0.1

        self.assertEqual(g1, g2)
        self.assertNotEqual(g1, g3)

        s = {g1, g3}
        self.assertIn(g2, s)
        self.assertEqual(len(s), 2)

    def test_distance_includes_weight_difference_and_enabled_flag(self):
        """Connection distance should account for weight difference and enabled mismatch."""
        cfg = _DummyConfig()

        g1 = DefaultConnectionGene((0, 1), innovation=1)
        g2 = DefaultConnectionGene((0, 1), innovation=1)

        g1.weight = 0.0
        g1.enabled = True
        g2.weight = 1.0
        g2.enabled = True

        # |0.0 - 1.0| = 1.0, scaled by 0.5.
        self.assertEqual(g1.distance(g2, cfg), 0.5)

        # Enabled mismatch adds 1.0 before scaling: (1.0 + 1.0) * 0.5 = 1.0.
        g2.enabled = False
        self.assertEqual(g1.distance(g2, cfg), 1.0)

    def test_copy_preserves_key_innovation_and_attributes(self):
        """BaseGene.copy should preserve key, innovation, and all attributes for connections."""
        g = DefaultConnectionGene((0, 1), innovation=42)
        g.weight = -0.75
        g.enabled = False

        clone = g.copy()
        self.assertIsNot(clone, g)
        self.assertIsInstance(clone, DefaultConnectionGene)
        self.assertEqual(clone.key, g.key)
        self.assertEqual(clone.innovation, g.innovation)
        self.assertEqual(clone.weight, g.weight)
        self.assertEqual(clone.enabled, g.enabled)

    def test_crossover_requires_matching_innovations_for_connection_genes(self):
        """Connection gene crossover should assert if innovation numbers disagree."""
        g1 = DefaultConnectionGene((0, 1), innovation=1)
        g2 = DefaultConnectionGene((0, 1), innovation=2)

        g1.weight = 0.5
        g1.enabled = True
        g2.weight = 0.5
        g2.enabled = True

        with self.assertRaises(AssertionError):
            g1.crossover(g2)

    def test_crossover_preserves_innovation_and_key(self):
        """Successful crossover preserves key and innovation on the offspring gene."""
        g1 = DefaultConnectionGene((0, 1), innovation=10)
        g2 = DefaultConnectionGene((0, 1), innovation=10)

        g1.weight = -1.0
        g1.enabled = True
        g2.weight = 2.0
        g2.enabled = False

        child = g1.crossover(g2)

        self.assertEqual(child.key, (0, 1))
        self.assertEqual(child.innovation, 10)
        # Weight and enabled should come from one of the parents.
        self.assertIn(child.weight, {-1.0, 2.0})
        self.assertIn(child.enabled, {True, False})


if __name__ == "__main__":
    unittest.main()
