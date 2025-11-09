"""
Tests for the 75% disable rule from the NEAT paper.

From Stanley & Miikkulainen (2002), p. 111:
"There was a 75% chance that an inherited gene was disabled if it was disabled in either parent."

Implementation Note:
The neat-python implementation applies the 75% rule AFTER random attribute inheritance.
For genes with one parent disabled and one enabled:
- The 'enabled' attribute is first randomly inherited (50/50)
- Then, if EITHER parent was disabled, there's a 75% chance to disable
- Effective rate: 50% (inherited disabled) + 50% * 75% (inherited enabled then disabled) = 87.5%

This implementation validates that the 75% rule is correctly applied as an additional
mechanism on top of standard attribute inheritance.
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
        """Test effective disable rate when one parent is disabled (~87.5%)."""
        gene1 = DefaultConnectionGene((0, 1), innovation=1)
        gene1.weight = 1.0
        gene1.enabled = False  # Disabled
        
        gene2 = DefaultConnectionGene((0, 1), innovation=1)
        gene2.weight = 1.0
        gene2.enabled = True   # Enabled
        
        trials = 2000
        disabled_count = sum(1 for _ in range(trials) 
                            if not gene1.crossover(gene2).enabled)
        
        disable_rate = disabled_count / trials
        
        # Effective rate should be ~87.5% (50% + 37.5%)
        self.assertGreater(disable_rate, 0.84, 
                          f"Expected ~87.5% effective disable rate, got {disable_rate:.2%}")
        self.assertLess(disable_rate, 0.91,
                       f"Expected ~87.5% effective disable rate, got {disable_rate:.2%}")
    
    def test_disable_rule_both_parents_disabled(self):
        """Test that offspring is always disabled when both parents are disabled."""
        gene1 = DefaultConnectionGene((0, 1), innovation=1)
        gene1.weight = 1.0
        gene1.enabled = False
        
        gene2 = DefaultConnectionGene((0, 1), innovation=1)
        gene2.weight = 2.0
        gene2.enabled = False
        
        trials = 1000
        disabled_count = sum(1 for _ in range(trials)
                            if not gene1.crossover(gene2).enabled)
        
        disable_rate = disabled_count / trials
        
        # Both disabled: 100% inherit disabled (since both parents are disabled)
        # The 75% rule doesn't change anything since the gene is already disabled
        self.assertEqual(disabled_count, trials,
                        f"Expected 100% disable rate with both parents disabled, got {disable_rate:.2%}")
    
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
        """Test that the 75% probability is correctly applied in the implementation."""
        # The key insight: enabled is first inherited randomly, 
        # THEN 75% rule is applied if either parent was disabled
        
        gene1 = DefaultConnectionGene((0, 1), innovation=1)
        gene1.weight = 1.0
        gene1.enabled = False
        
        gene2 = DefaultConnectionGene((0, 1), innovation=1)
        gene2.weight = 1.0
        gene2.enabled = True
        
        trials = 5000
        disabled_count = sum(1 for _ in range(trials)
                            if not gene1.crossover(gene2).enabled)
        
        disable_rate = disabled_count / trials
        
        # Should NOT be exactly 75%
        self.assertNotAlmostEqual(disable_rate, 0.75, delta=0.03,
                                 msg="Rate should not be exactly 75% due to layered inheritance")
        
        # Should be close to 87.5%
        self.assertAlmostEqual(disable_rate, 0.875, delta=0.03,
                              msg=f"Expected ~87.5% from layered inheritance, got {disable_rate:.3f}")
    
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
        
        trials = 2000
        
        # Crossover both ways
        disabled_count_1 = sum(1 for _ in range(trials) 
                               if not gene1_disabled.crossover(gene2_enabled).enabled)
        disabled_count_2 = sum(1 for _ in range(trials)
                               if not gene2_enabled.crossover(gene1_disabled).enabled)
        
        rate1 = disabled_count_1 / trials
        rate2 = disabled_count_2 / trials
        
        # Both should be approximately the same (~87.5%)
        self.assertAlmostEqual(rate1, rate2, delta=0.05,
                              msg=f"Rates should be similar: {rate1:.2%} vs {rate2:.2%}")


class TestDisableRuleImplementation(unittest.TestCase):
    """Tests that verify the implementation details of the disable rule."""
    
    def test_rule_applied_after_attribute_inheritance(self):
        """Verify that the 75% rule is applied AFTER random attribute inheritance."""
        # This is the key implementation detail
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
        
        # If rule was applied INSTEAD of inheritance, we'd get exactly 75%
        # Since it's applied AFTER, we get 87.5%
        self.assertGreater(rate, 0.85, 
                          "Rate should be higher than 75% due to layered application")
        self.assertLess(rate, 0.90,
                       "Rate should be close to 87.5%")
    
    def test_enabled_offspring_probability(self):
        """Test that ~12.5% of offspring remain enabled (complement of 87.5%)."""
        gene1 = DefaultConnectionGene((0, 1), innovation=1)
        gene1.weight = 1.0
        gene1.enabled = False
        
        gene2 = DefaultConnectionGene((0, 1), innovation=1)
        gene2.weight = 1.0
        gene2.enabled = True
        
        trials = 2000
        enabled_count = sum(1 for _ in range(trials)
                           if gene1.crossover(gene2).enabled)
        
        enabled_rate = enabled_count / trials
        
        # Should have ~12.5% enabled (50% inherit enabled * 25% stay enabled)
        self.assertGreater(enabled_rate, 0.09,
                          f"Expected ~12.5% enabled, got {enabled_rate:.2%}")
        self.assertLess(enabled_rate, 0.16,
                       f"Expected ~12.5% enabled, got {enabled_rate:.2%}")
    
    def test_no_false_enabling(self):
        """Test that the rule never RE-enables an already disabled gene."""
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
        # P(disabled) = P(inherit disabled) + P(inherit enabled AND then disabled)
        #             = 0.5 + (0.5 * 0.75)
        #             = 0.5 + 0.375
        #             = 0.875
        expected_rate = 0.875
        
        # Should be within 1% of expected
        self.assertAlmostEqual(observed_rate, expected_rate, delta=0.01,
                              msg=f"Observed {observed_rate:.4f} should match model {expected_rate}")


if __name__ == '__main__':
    unittest.main()
