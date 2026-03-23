"""
Tests for the 75% disable rule from the NEAT paper.

From Stanley & Miikkulainen (2002), p. 111:
"There was a 75% chance that an inherited gene was disabled if it was disabled in either parent."

Implementation:
When either parent has the gene disabled, the randomly-inherited enabled attribute is
REPLACED by a fresh 75/25 coin flip (75% disabled, 25% enabled). This applies regardless
of which value was inherited from the random attribute selection step.
"""
import os
import unittest

import neat
from neat.genes import DefaultConnectionGene


class Test75PercentDisableRule(unittest.TestCase):
    """Tests for the 75% disable rule during crossover."""

    def setUp(self):
        """Set up test configuration."""
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)

    def test_disable_rule_one_parent_disabled(self):
        """Test effective disable rate when one parent is disabled (75%)."""
        gene1 = DefaultConnectionGene((0, 1), innovation=1)
        gene1.weight = 1.0
        gene1.enabled = False  # Disabled

        gene2 = DefaultConnectionGene((0, 1), innovation=1)
        gene2.weight = 1.0
        gene2.enabled = True   # Enabled

        trials = 5000
        disabled_count = sum(1 for _ in range(trials)
                            if not gene1.crossover(gene2).enabled)

        disable_rate = disabled_count / trials

        # Effective rate should be 75% (the rule replaces inherited value)
        self.assertAlmostEqual(disable_rate, 0.75, delta=0.03,
                              msg=f"Expected ~75% disable rate, got {disable_rate:.2%}")

    def test_disable_rule_both_parents_disabled(self):
        """Test disable rate when both parents are disabled (75%)."""
        gene1 = DefaultConnectionGene((0, 1), innovation=1)
        gene1.weight = 1.0
        gene1.enabled = False

        gene2 = DefaultConnectionGene((0, 1), innovation=1)
        gene2.weight = 2.0
        gene2.enabled = False

        trials = 5000
        disabled_count = sum(1 for _ in range(trials)
                            if not gene1.crossover(gene2).enabled)

        disable_rate = disabled_count / trials

        # Both disabled: the 75% rule replaces inherited value → 75% disabled
        self.assertAlmostEqual(disable_rate, 0.75, delta=0.03,
                              msg=f"Expected ~75% disable rate, got {disable_rate:.2%}")

    def test_disable_rule_both_parents_enabled(self):
        """Test that offspring is always enabled when both parents are enabled."""
        gene1 = DefaultConnectionGene((0, 1), innovation=1)
        gene1.weight = 1.0
        gene1.enabled = True

        gene2 = DefaultConnectionGene((0, 1), innovation=1)
        gene2.weight = 2.0
        gene2.enabled = True

        trials = 1000
        disabled_count = sum(1 for _ in range(trials)
                            if not gene1.crossover(gene2).enabled)

        # Should be 0% disabled (all enabled) since neither parent is disabled
        self.assertEqual(disabled_count, 0,
                        f"Expected 0% disable rate with both parents enabled, got {disabled_count/trials:.2%}")

    def test_disable_rule_is_applied_correctly(self):
        """Test that the 75% probability directly replaces inherited enabled value."""
        gene1 = DefaultConnectionGene((0, 1), innovation=1)
        gene1.weight = 1.0
        gene1.enabled = False

        gene2 = DefaultConnectionGene((0, 1), innovation=1)
        gene2.weight = 1.0
        gene2.enabled = True

        trials = 10000
        disabled_count = sum(1 for _ in range(trials)
                            if not gene1.crossover(gene2).enabled)

        disable_rate = disabled_count / trials

        # Should be exactly 75% (rule replaces inherited value directly)
        self.assertAlmostEqual(disable_rate, 0.75, delta=0.02,
                              msg=f"Expected ~75% from direct replacement, got {disable_rate:.3f}")

    def test_disable_rule_preserves_other_attributes(self):
        """Test that the disable rule doesn't affect other attribute inheritance."""
        gene1 = DefaultConnectionGene((0, 1), innovation=1)
        gene1.weight = 1.0
        gene1.enabled = False

        gene2 = DefaultConnectionGene((0, 1), innovation=1)
        gene2.weight = 2.0
        gene2.enabled = True

        for _ in range(100):
            offspring = gene1.crossover(gene2)

            # Weight should be from one of the parents
            self.assertIn(offspring.weight, [1.0, 2.0],
                         "Offspring weight should be from one of the parents")

            # Innovation number should be preserved
            self.assertEqual(offspring.innovation, 1,
                           "Innovation number should be preserved")

    def test_disable_rule_symmetry(self):
        """Test that disable rule works the same regardless of parent order."""
        gene1_disabled = DefaultConnectionGene((0, 1), innovation=1)
        gene1_disabled.weight = 1.0
        gene1_disabled.enabled = False

        gene2_enabled = DefaultConnectionGene((0, 1), innovation=1)
        gene2_enabled.weight = 1.0
        gene2_enabled.enabled = True

        trials = 5000

        # Crossover both ways
        disabled_count_1 = sum(1 for _ in range(trials)
                               if not gene1_disabled.crossover(gene2_enabled).enabled)
        disabled_count_2 = sum(1 for _ in range(trials)
                               if not gene2_enabled.crossover(gene1_disabled).enabled)

        rate1 = disabled_count_1 / trials
        rate2 = disabled_count_2 / trials

        # Both should be approximately the same (~75%)
        self.assertAlmostEqual(rate1, rate2, delta=0.05,
                              msg=f"Rates should be similar: {rate1:.2%} vs {rate2:.2%}")
        self.assertAlmostEqual(rate1, 0.75, delta=0.03)
        self.assertAlmostEqual(rate2, 0.75, delta=0.03)


class TestDisableRuleImplementation(unittest.TestCase):
    """Tests that verify the implementation details of the disable rule."""

    def test_rule_replaces_inherited_value(self):
        """Verify that the 75% rule replaces (not layers on) inherited enabled value."""
        gene1 = DefaultConnectionGene((0, 1), innovation=1)
        gene1.weight = 1.0
        gene1.enabled = False

        gene2 = DefaultConnectionGene((0, 1), innovation=1)
        gene2.weight = 2.0
        gene2.enabled = True

        trials = 10000
        disabled_count = sum(1 for _ in range(trials)
                            if not gene1.crossover(gene2).enabled)

        rate = disabled_count / trials

        # If rule was layered (old behavior), we'd get ~87.5%
        # Since it replaces, we get exactly 75%
        self.assertAlmostEqual(rate, 0.75, delta=0.02,
                              msg=f"Rate should be 75%, not 87.5%. Got {rate:.3f}")
        # Explicitly verify it's NOT the old layered rate
        self.assertLess(rate, 0.83,
                       "Rate should be well below the old layered 87.5%")

    def test_enabled_offspring_probability(self):
        """Test that ~25% of offspring remain enabled (complement of 75%)."""
        gene1 = DefaultConnectionGene((0, 1), innovation=1)
        gene1.weight = 1.0
        gene1.enabled = False

        gene2 = DefaultConnectionGene((0, 1), innovation=1)
        gene2.weight = 1.0
        gene2.enabled = True

        trials = 5000
        enabled_count = sum(1 for _ in range(trials)
                           if gene1.crossover(gene2).enabled)

        enabled_rate = enabled_count / trials

        # Should have ~25% enabled
        self.assertAlmostEqual(enabled_rate, 0.25, delta=0.03,
                              msg=f"Expected ~25% enabled, got {enabled_rate:.2%}")

    def test_no_false_disabling(self):
        """Test that the rule never disables when both parents are enabled."""
        gene1 = DefaultConnectionGene((0, 1), innovation=1)
        gene1.weight = 1.0
        gene1.enabled = True

        gene2 = DefaultConnectionGene((0, 1), innovation=1)
        gene2.weight = 1.0
        gene2.enabled = True

        # When both enabled, should never get disabled
        for _ in range(1000):
            offspring = gene1.crossover(gene2)
            self.assertTrue(offspring.enabled,
                          "Should never disable when both parents enabled")

    def test_mathematical_model_accuracy(self):
        """Test that observed rates match the mathematical model."""
        gene1 = DefaultConnectionGene((0, 1), innovation=1)
        gene1.weight = 1.0
        gene1.enabled = False

        gene2 = DefaultConnectionGene((0, 1), innovation=1)
        gene2.weight = 1.0
        gene2.enabled = True

        trials = 10000
        disabled_count = sum(1 for _ in range(trials)
                            if not gene1.crossover(gene2).enabled)

        observed_rate = disabled_count / trials

        # Mathematical model:
        # When either parent disabled, inherited enabled value is replaced:
        # P(disabled) = 0.75  (direct coin flip)
        expected_rate = 0.75

        # Should be within 1% of expected
        self.assertAlmostEqual(observed_rate, expected_rate, delta=0.01,
                              msg=f"Observed {observed_rate:.4f} should match model {expected_rate}")


if __name__ == '__main__':
    unittest.main()
