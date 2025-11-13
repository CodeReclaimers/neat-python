"""
Tests for GitHub issue #188: Class attribute bug in neat.attributes

This tests that _config_items is properly isolated between instances
rather than being shared at the class level.
"""

import unittest
from neat.attributes import StringAttribute, FloatAttribute, BoolAttribute, IntegerAttribute


class TestIssue188AttributeIsolation(unittest.TestCase):
    """Test that attribute instances don't share _config_items."""

    def test_string_attribute_isolation(self):
        """Test that StringAttribute instances have separate _config_items."""
        attr1 = StringAttribute('activation', options='sigmoid')
        attr2 = StringAttribute('aggregation', options='sum')

        # Should NOT share the same object
        self.assertIsNot(attr1._config_items, attr2._config_items,
                        "StringAttribute instances should not share _config_items")

        # Should have different option values
        self.assertEqual(attr1._config_items['options'][1], 'sigmoid')
        self.assertEqual(attr2._config_items['options'][1], 'sum')

    def test_string_attribute_mutation_isolation(self):
        """Test that modifying one instance doesn't affect another."""
        attr1 = StringAttribute('foo', options='option_a')
        attr2 = StringAttribute('bar', options='option_b')

        # Modify attr1's _config_items
        attr1._config_items['options'] = ['MODIFIED', 'TEST']

        # attr2 should NOT be affected
        self.assertEqual(attr2._config_items['options'][1], 'option_b',
                        "Modifying attr1 should not affect attr2")

    def test_float_attribute_isolation(self):
        """Test that FloatAttribute instances have separate _config_items."""
        attr1 = FloatAttribute('weight', init_mean=0.0)
        attr2 = FloatAttribute('bias', init_mean=1.0)

        self.assertIsNot(attr1._config_items, attr2._config_items)
        self.assertEqual(attr1._config_items['init_mean'][1], 0.0)
        self.assertEqual(attr2._config_items['init_mean'][1], 1.0)

    def test_bool_attribute_isolation(self):
        """Test that BoolAttribute instances have separate _config_items."""
        attr1 = BoolAttribute('enabled', default='True')
        attr2 = BoolAttribute('flag', default='False')

        self.assertIsNot(attr1._config_items, attr2._config_items)
        self.assertEqual(attr1._config_items['default'][1], 'True')
        self.assertEqual(attr2._config_items['default'][1], 'False')

    def test_integer_attribute_isolation(self):
        """Test that IntegerAttribute instances have separate _config_items."""
        attr1 = IntegerAttribute('count', min_value=0)
        attr2 = IntegerAttribute('index', min_value=1)

        self.assertIsNot(attr1._config_items, attr2._config_items)
        self.assertEqual(attr1._config_items['min_value'][1], 0)
        self.assertEqual(attr2._config_items['min_value'][1], 1)

    def test_mixed_attribute_types_isolation(self):
        """Test that different attribute types don't interfere with each other."""
        float_attr = FloatAttribute('weight')
        string_attr = StringAttribute('activation')
        bool_attr = BoolAttribute('enabled')
        int_attr = IntegerAttribute('count')

        # Each should have its own _config_items
        items_list = [
            float_attr._config_items,
            string_attr._config_items,
            bool_attr._config_items,
            int_attr._config_items
        ]

        # All should be different objects
        for i, items_i in enumerate(items_list):
            for j, items_j in enumerate(items_list):
                if i != j:
                    self.assertIsNot(items_i, items_j,
                                   f"Attribute instances {i} and {j} should not share _config_items")

    def test_genes_pattern(self):
        """Test the actual usage pattern from DefaultNodeGene."""
        # This is how it's used in genes.py
        attr_activation = StringAttribute('activation', options='')
        attr_aggregation = StringAttribute('aggregation', options='')

        # Should have separate _config_items even with same default value
        self.assertIsNot(attr_activation._config_items, attr_aggregation._config_items,
                        "activation and aggregation attributes should have separate _config_items")

        # Both should have empty string as options default
        self.assertEqual(attr_activation._config_items['options'][1], '')
        self.assertEqual(attr_aggregation._config_items['options'][1], '')

        # Modifying one should not affect the other
        attr_activation._config_items['options'][1] = 'modified'
        self.assertEqual(attr_aggregation._config_items['options'][1], '',
                        "Modifying activation should not affect aggregation")


if __name__ == '__main__':
    unittest.main()
