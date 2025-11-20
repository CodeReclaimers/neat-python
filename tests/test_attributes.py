"""
Comprehensive tests for attribute system in neat-python.

Tests cover:
- FloatAttribute: Gaussian/uniform initialization, mutation, replacement, clamping
- IntegerAttribute: Initialization, mutation, replacement, clamping
- BoolAttribute: Initialization with various defaults, mutation with bias
- StringAttribute: Random selection, mutation to different options
- Attribute validation: Invalid configurations, boundary conditions
- Edge cases: Zero mutation rates, rate=1.0, extreme values

"""

import unittest
from neat.attributes import FloatAttribute, IntegerAttribute, BoolAttribute, StringAttribute


class MockConfig:
    """Mock configuration object for testing attributes."""
    pass


def create_float_config(attr_name, init_mean=0.0, init_stdev=1.0, init_type='gaussian',
                       replace_rate=0.0, mutate_rate=0.0, mutate_power=0.5,
                       min_value=-30.0, max_value=30.0):
    """Create a mock config for FloatAttribute testing."""
    config = MockConfig()
    setattr(config, f'{attr_name}_init_mean', init_mean)
    setattr(config, f'{attr_name}_init_stdev', init_stdev)
    setattr(config, f'{attr_name}_init_type', init_type)
    setattr(config, f'{attr_name}_replace_rate', replace_rate)
    setattr(config, f'{attr_name}_mutate_rate', mutate_rate)
    setattr(config, f'{attr_name}_mutate_power', mutate_power)
    setattr(config, f'{attr_name}_min_value', min_value)
    setattr(config, f'{attr_name}_max_value', max_value)
    return config


def create_integer_config(attr_name, replace_rate=0.0, mutate_rate=0.0, 
                         mutate_power=0.5, min_value=-30, max_value=30):
    """Create a mock config for IntegerAttribute testing."""
    config = MockConfig()
    setattr(config, f'{attr_name}_replace_rate', replace_rate)
    setattr(config, f'{attr_name}_mutate_rate', mutate_rate)
    setattr(config, f'{attr_name}_mutate_power', mutate_power)
    setattr(config, f'{attr_name}_min_value', min_value)
    setattr(config, f'{attr_name}_max_value', max_value)
    return config


def create_bool_config(attr_name, default='random', mutate_rate=0.0,
                      rate_to_true_add=0.0, rate_to_false_add=0.0):
    """Create a mock config for BoolAttribute testing."""
    config = MockConfig()
    setattr(config, f'{attr_name}_default', default)
    setattr(config, f'{attr_name}_mutate_rate', mutate_rate)
    setattr(config, f'{attr_name}_rate_to_true_add', rate_to_true_add)
    setattr(config, f'{attr_name}_rate_to_false_add', rate_to_false_add)
    return config


def create_string_config(attr_name, default='random', options=None, mutate_rate=0.0):
    """Create a mock config for StringAttribute testing."""
    config = MockConfig()
    if options is None:
        options = ['option1', 'option2', 'option3']
    setattr(config, f'{attr_name}_default', default)
    setattr(config, f'{attr_name}_options', options)
    setattr(config, f'{attr_name}_mutate_rate', mutate_rate)
    return config


class TestFloatAttribute(unittest.TestCase):
    """Tests for FloatAttribute initialization, mutation, and clamping."""
    
    def test_gaussian_initialization(self):
        """
        Test that Gaussian initialization produces values within expected range.
        
        With Gaussian distribution, values should cluster around the mean
        and respect min/max bounds.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', init_mean=5.0, init_stdev=2.0,
                                     init_type='gaussian', min_value=0.0, max_value=10.0)
        
        # Generate many values and check they're within bounds
        values = [attr.init_value(config) for _ in range(1000)]
        
        for v in values:
            self.assertGreaterEqual(v, 0.0, "Value should be >= min_value")
            self.assertLessEqual(v, 10.0, "Value should be <= max_value")
        
        # Check mean is approximately correct
        mean = sum(values) / len(values)
        self.assertAlmostEqual(mean, 5.0, delta=0.5, 
                              msg="Mean should be close to init_mean")
    
    def test_uniform_initialization(self):
        """
        Test that uniform initialization produces evenly distributed values.
        
        With uniform distribution, values should be spread across the range
        defined by mean ± 2*stdev, clamped to min/max.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', init_mean=5.0, init_stdev=2.0,
                                     init_type='uniform', min_value=0.0, max_value=10.0)
        
        # Generate many values
        values = [attr.init_value(config) for _ in range(1000)]
        
        # All should be in range [max(0, 5-4), min(10, 5+4)] = [1, 9]
        for v in values:
            self.assertGreaterEqual(v, 1.0)
            self.assertLessEqual(v, 9.0)
        
        # Check distribution is reasonably uniform (not clustered at center)
        lower_half = sum(1 for v in values if v < 5.0)
        upper_half = sum(1 for v in values if v >= 5.0)
        # Should be roughly 50/50, allow some variance
        self.assertGreater(lower_half, 400)
        self.assertGreater(upper_half, 400)
    
    def test_clamping_to_bounds(self):
        """
        Test that clamp() properly enforces min/max bounds.
        
        Values outside the configured range should be clamped to the boundaries.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', min_value=-10.0, max_value=10.0)
        
        # Test clamping below minimum
        self.assertEqual(attr.clamp(-15.0, config), -10.0)
        
        # Test clamping above maximum
        self.assertEqual(attr.clamp(15.0, config), 10.0)
        
        # Test value within range (no clamping)
        self.assertEqual(attr.clamp(5.0, config), 5.0)
        
        # Test boundary values
        self.assertEqual(attr.clamp(-10.0, config), -10.0)
        self.assertEqual(attr.clamp(10.0, config), 10.0)
    
    def test_mutation_vs_replacement(self):
        """
        Test that mutation and replacement occur at correct rates.
        
        Mutation should add Gaussian noise, while replacement should
        generate a completely new value.
        """
        attr = FloatAttribute('test_attr')
        
        # Test pure mutation (no replacement)
        config_mutate = create_float_config('test_attr', init_mean=0.0, init_stdev=1.0,
                                           mutate_rate=1.0, replace_rate=0.0,
                                           mutate_power=0.1, min_value=-100.0, max_value=100.0)
        
        initial_value = 10.0
        mutated_values = [attr.mutate_value(initial_value, config_mutate) for _ in range(100)]
        
        # All mutated values should be close to initial (small mutate_power)
        for v in mutated_values:
            self.assertLess(abs(v - initial_value), 5.0, 
                          "Mutated value should be close to initial with small mutate_power")
        
        # Test replacement (should generate values near init_mean=0)
        config_replace = create_float_config('test_attr', init_mean=0.0, init_stdev=1.0,
                                            mutate_rate=0.0, replace_rate=1.0,
                                            mutate_power=0.1, min_value=-100.0, max_value=100.0)
        
        replaced_values = [attr.mutate_value(initial_value, config_replace) for _ in range(100)]
        
        # Replaced values should cluster around init_mean (0), not initial value (10)
        mean_replaced = sum(replaced_values) / len(replaced_values)
        self.assertLess(abs(mean_replaced), 2.0, 
                       "Replaced values should cluster around init_mean, not initial value")
    
    def test_zero_mutation_rate(self):
        """
        Test that zero mutation rate leaves value unchanged.
        
        Edge case: with mutate_rate=0 and replace_rate=0, value should never change.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', mutate_rate=0.0, replace_rate=0.0)
        
        initial_value = 5.0
        for _ in range(100):
            result = attr.mutate_value(initial_value, config)
            self.assertEqual(result, initial_value, 
                           "Value should not change with zero mutation rate")
    
    def test_mutation_rate_one(self):
        """
        Test that mutation_rate=1.0 always mutates the value.
        
        Edge case: with rate=1.0, value should change every time.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', mutate_rate=1.0, replace_rate=0.0,
                                     mutate_power=1.0, min_value=-100.0, max_value=100.0)
        
        initial_value = 5.0
        changed_count = 0
        for _ in range(100):
            result = attr.mutate_value(initial_value, config)
            if result != initial_value:
                changed_count += 1
        
        # With mutate_power=1.0, most mutations should change the value
        self.assertGreater(changed_count, 90, 
                         "With rate=1.0, value should change in most cases")
    
    def test_very_small_mutate_power(self):
        """
        Test mutation with very small mutate_power.
        
        Edge case: small power should produce tiny mutations.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', mutate_rate=1.0, replace_rate=0.0,
                                     mutate_power=0.001, min_value=-100.0, max_value=100.0)
        
        initial_value = 10.0
        values = [attr.mutate_value(initial_value, config) for _ in range(100)]
        
        # All values should be very close to initial
        for v in values:
            self.assertLess(abs(v - initial_value), 0.1, 
                          "Small mutate_power should produce tiny changes")
    
    def test_very_large_mutate_power(self):
        """
        Test mutation with very large mutate_power.
        
        Edge case: large power should produce large mutations, respecting bounds.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', mutate_rate=1.0, replace_rate=0.0,
                                     mutate_power=50.0, min_value=-30.0, max_value=30.0)
        
        initial_value = 0.0
        values = [attr.mutate_value(initial_value, config) for _ in range(100)]
        
        # Values should be spread across the range, all within bounds
        for v in values:
            self.assertGreaterEqual(v, -30.0)
            self.assertLessEqual(v, 30.0)
        
        # Should have significant variance
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        self.assertGreater(variance, 10.0, 
                         "Large mutate_power should produce high variance")
    
    def test_validation_invalid_min_max(self):
        """
        Test that validation catches invalid min/max configuration.
        
        Configuration with max < min should raise RuntimeError.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', min_value=10.0, max_value=5.0)
        
        with self.assertRaises(RuntimeError):
            attr.validate(config)
    
    def test_validation_valid_min_max(self):
        """
        Test that validation passes with valid min/max configuration.
        
        Configuration with max >= min should not raise an error.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', min_value=5.0, max_value=10.0)
        
        # Should not raise an exception
        attr.validate(config)
    
    def test_init_value_clamping(self):
        """
        Test that init_value respects clamping even for extreme Gaussian values.
        
        Even with large stdev, initialized values should be clamped to bounds.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', init_mean=0.0, init_stdev=1000.0,
                                     init_type='gaussian', min_value=-10.0, max_value=10.0)
        
        # Generate many values - all should be clamped
        values = [attr.init_value(config) for _ in range(100)]
        
        for v in values:
            self.assertGreaterEqual(v, -10.0)
            self.assertLessEqual(v, 10.0)


class TestIntegerAttribute(unittest.TestCase):
    """Tests for IntegerAttribute initialization, mutation, and clamping."""
    
    def test_integer_initialization(self):
        """
        Test that initialization produces integers within the specified range.
        
        Random integers should be evenly distributed between min and max.
        """
        attr = IntegerAttribute('test_attr')
        config = create_integer_config('test_attr', min_value=0, max_value=10)
        
        values = [attr.init_value(config) for _ in range(1000)]
        
        # All should be integers in range
        for v in values:
            self.assertIsInstance(v, int)
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 10)
        
        # Should have decent coverage of the range
        unique_values = set(values)
        self.assertGreater(len(unique_values), 5, 
                         "Should generate diverse integer values")
    
    def test_integer_clamping(self):
        """
        Test that clamp() properly enforces integer bounds.
        
        Values outside the configured range should be clamped to boundaries.
        """
        attr = IntegerAttribute('test_attr')
        config = create_integer_config('test_attr', min_value=-10, max_value=10)
        
        # Test clamping
        self.assertEqual(attr.clamp(-15, config), -10)
        self.assertEqual(attr.clamp(15, config), 10)
        self.assertEqual(attr.clamp(5, config), 5)
        self.assertEqual(attr.clamp(-10, config), -10)
        self.assertEqual(attr.clamp(10, config), 10)
    
    def test_integer_mutation(self):
        """
        Test that mutation adds integer offsets to values.
        
        Mutation should add rounded Gaussian noise as an integer offset.
        """
        attr = IntegerAttribute('test_attr')
        config = create_integer_config('test_attr', mutate_rate=1.0, replace_rate=0.0,
                                      mutate_power=2.0, min_value=-100, max_value=100)
        
        initial_value = 10
        values = [attr.mutate_value(initial_value, config) for _ in range(100)]
        
        # All should be integers
        for v in values:
            self.assertIsInstance(v, int)
        
        # Values should be close to initial (mutate_power=2.0)
        for v in values:
            self.assertLess(abs(v - initial_value), 15, 
                          "Mutated integers should be close to initial")
    
    def test_integer_replacement(self):
        """
        Test that replacement generates new random integers.
        
        Replacement should produce values across the entire range.
        """
        attr = IntegerAttribute('test_attr')
        config = create_integer_config('test_attr', mutate_rate=0.0, replace_rate=1.0,
                                      min_value=0, max_value=100)
        
        initial_value = 10
        values = [attr.mutate_value(initial_value, config) for _ in range(100)]
        
        # Should generate diverse values across range
        unique_values = set(values)
        self.assertGreater(len(unique_values), 20, 
                         "Replacement should generate diverse values")
        
        # Mean should be around 50 (middle of 0-100)
        mean = sum(values) / len(values)
        self.assertLess(abs(mean - 50), 15, 
                       "Mean of replaced values should be around range midpoint")
    
    def test_integer_validation_invalid_min_max(self):
        """
        Test that validation catches invalid min/max configuration for integers.
        """
        attr = IntegerAttribute('test_attr')
        config = create_integer_config('test_attr', min_value=10, max_value=5)
        
        with self.assertRaises(RuntimeError):
            attr.validate(config)
    
    def test_integer_zero_mutation_rate(self):
        """
        Test that zero mutation rate leaves integer value unchanged.
        """
        attr = IntegerAttribute('test_attr')
        config = create_integer_config('test_attr', mutate_rate=0.0, replace_rate=0.0)
        
        initial_value = 5
        for _ in range(100):
            result = attr.mutate_value(initial_value, config)
            self.assertEqual(result, initial_value)


class TestBoolAttribute(unittest.TestCase):
    """Tests for BoolAttribute initialization and mutation with bias."""
    
    def test_bool_initialization_true(self):
        """
        Test initialization with default='true'.
        
        Various string representations of true should all initialize to True.
        """
        attr = BoolAttribute('test_attr')
        
        for true_val in ['true', 'True', 'TRUE', '1', 'yes', 'on']:
            config = create_bool_config('test_attr', default=true_val)
            value = attr.init_value(config)
            self.assertTrue(value, f"'{true_val}' should initialize to True")
    
    def test_bool_initialization_false(self):
        """
        Test initialization with default='false'.
        
        Various string representations of false should all initialize to False.
        """
        attr = BoolAttribute('test_attr')
        
        for false_val in ['false', 'False', 'FALSE', '0', 'no', 'off']:
            config = create_bool_config('test_attr', default=false_val)
            value = attr.init_value(config)
            self.assertFalse(value, f"'{false_val}' should initialize to False")
    
    def test_bool_initialization_random(self):
        """
        Test initialization with default='random'.
        
        Random initialization should produce both True and False values.
        """
        attr = BoolAttribute('test_attr')
        config = create_bool_config('test_attr', default='random')
        
        values = [attr.init_value(config) for _ in range(100)]
        
        # Should have both True and False
        self.assertIn(True, values, "Random init should produce True")
        self.assertIn(False, values, "Random init should produce False")
        
        # Should be roughly 50/50
        true_count = sum(values)
        self.assertGreater(true_count, 30, "Should have significant True values")
        self.assertLess(true_count, 70, "Should have significant False values")
    
    def test_bool_mutation_with_rate_to_true_add(self):
        """
        Test mutation bias toward True with rate_to_true_add.
        
        When value is False, additional rate should increase mutation probability.
        """
        attr = BoolAttribute('test_attr')
        config = create_bool_config('test_attr', mutate_rate=0.5, rate_to_true_add=0.5)
        
        # Starting from False, effective rate = 0.5 + 0.5 = 1.0
        # Should mutate frequently
        initial_value = False
        mutations = [attr.mutate_value(initial_value, config) for _ in range(100)]
        
        # With effective rate of 1.0, should get many Trues (random 50/50 when mutating)
        true_count = sum(mutations)
        # Even with random choice, should get some Trues
        self.assertGreater(true_count, 0, 
                         "High mutation rate from False should produce some True values")
    
    def test_bool_mutation_with_rate_to_false_add(self):
        """
        Test mutation bias toward False with rate_to_false_add.
        
        When value is True, additional rate should increase mutation probability.
        """
        attr = BoolAttribute('test_attr')
        config = create_bool_config('test_attr', mutate_rate=0.5, rate_to_false_add=0.5)
        
        # Starting from True, effective rate = 0.5 + 0.5 = 1.0
        initial_value = True
        mutations = [attr.mutate_value(initial_value, config) for _ in range(100)]
        
        # With effective rate of 1.0, should get many Falses (random 50/50 when mutating)
        false_count = sum(1 for v in mutations if not v)
        self.assertGreater(false_count, 0, 
                         "High mutation rate from True should produce some False values")
    
    def test_bool_zero_mutation_rate(self):
        """
        Test that zero mutation rate leaves boolean value unchanged.
        """
        attr = BoolAttribute('test_attr')
        config = create_bool_config('test_attr', mutate_rate=0.0)
        
        # Test both True and False
        for initial_value in [True, False]:
            for _ in range(100):
                result = attr.mutate_value(initial_value, config)
                self.assertEqual(result, initial_value, 
                               "Value should not change with zero mutation rate")
    
    def test_bool_mutation_returns_random_value(self):
        """
        Test that mutation returns a random value (not guaranteed flip).
        
        The mutation may return the same value - it's a random choice.
        """
        attr = BoolAttribute('test_attr')
        config = create_bool_config('test_attr', mutate_rate=1.0)
        
        # With rate=1.0, always mutates but result is random
        initial_value = True
        results = [attr.mutate_value(initial_value, config) for _ in range(100)]
        
        # Should have both True and False in results
        self.assertIn(True, results)
        self.assertIn(False, results)
    
    def test_bool_validation_invalid_default(self):
        """
        Test that validation catches invalid default values.
        """
        attr = BoolAttribute('test_attr')
        config = create_bool_config('test_attr', default='invalid_value')
        
        with self.assertRaises(RuntimeError):
            attr.validate(config)
    
    def test_bool_validation_valid_defaults(self):
        """
        Test that validation passes with valid default values.
        """
        attr = BoolAttribute('test_attr')
        
        valid_defaults = ['true', 'false', 'random', 'none', '1', '0', 'yes', 'no', 'on', 'off']
        for default in valid_defaults:
            config = create_bool_config('test_attr', default=default)
            # Should not raise an exception
            attr.validate(config)


class TestStringAttribute(unittest.TestCase):
    """Tests for StringAttribute selection and mutation."""
    
    def test_string_initialization_specific_default(self):
        """
        Test initialization with a specific default value.
        
        When default is one of the options, it should be used.
        """
        attr = StringAttribute('test_attr')
        options = ['sigmoid', 'tanh', 'relu']
        config = create_string_config('test_attr', default='sigmoid', options=options)
        
        value = attr.init_value(config)
        self.assertEqual(value, 'sigmoid')
    
    def test_string_initialization_random(self):
        """
        Test initialization with default='random'.
        
        Should randomly select from available options.
        """
        attr = StringAttribute('test_attr')
        options = ['sigmoid', 'tanh', 'relu']
        config = create_string_config('test_attr', default='random', options=options)
        
        values = [attr.init_value(config) for _ in range(100)]
        
        # All values should be from options
        for v in values:
            self.assertIn(v, options)
        
        # Should use multiple options
        unique_values = set(values)
        self.assertGreater(len(unique_values), 1, 
                         "Random init should use multiple options")
    
    def test_string_initialization_none(self):
        """
        Test initialization with default='none'.
        
        'none' should be treated like 'random'.
        """
        attr = StringAttribute('test_attr')
        options = ['option1', 'option2', 'option3']
        config = create_string_config('test_attr', default='none', options=options)
        
        values = [attr.init_value(config) for _ in range(100)]
        
        # Should use multiple options
        unique_values = set(values)
        self.assertGreater(len(unique_values), 1, 
                         "'none' should work like 'random'")
    
    def test_string_mutation_changes_value(self):
        """
        Test that mutation can change the string value.
        
        With mutation, should randomly select from options.
        """
        attr = StringAttribute('test_attr')
        options = ['sigmoid', 'tanh', 'relu', 'sin', 'cos']
        config = create_string_config('test_attr', default='sigmoid', 
                                     options=options, mutate_rate=1.0)
        
        initial_value = 'sigmoid'
        values = [attr.mutate_value(initial_value, config) for _ in range(100)]
        
        # All should be from options
        for v in values:
            self.assertIn(v, options)
        
        # Should get some different values (mutation doesn't guarantee change)
        unique_values = set(values)
        self.assertGreater(len(unique_values), 1, 
                         "Mutation should produce diverse values")
    
    def test_string_zero_mutation_rate(self):
        """
        Test that zero mutation rate leaves string value unchanged.
        """
        attr = StringAttribute('test_attr')
        options = ['option1', 'option2', 'option3']
        config = create_string_config('test_attr', options=options, mutate_rate=0.0)
        
        initial_value = 'option1'
        for _ in range(100):
            result = attr.mutate_value(initial_value, config)
            self.assertEqual(result, initial_value, 
                           "Value should not change with zero mutation rate")
    
    def test_string_validation_invalid_default(self):
        """
        Test that validation catches invalid default values.
        
        Default must be either 'random'/'none' or one of the options.
        """
        attr = StringAttribute('test_attr')
        options = ['sigmoid', 'tanh', 'relu']
        config = create_string_config('test_attr', default='invalid_option', options=options)
        
        with self.assertRaises(RuntimeError):
            attr.validate(config)
    
    def test_string_validation_valid_defaults(self):
        """
        Test that validation passes with valid default values.
        """
        attr = StringAttribute('test_attr')
        options = ['sigmoid', 'tanh', 'relu']
        
        # Test with option from list
        config1 = create_string_config('test_attr', default='sigmoid', options=options)
        attr.validate(config1)
        
        # Test with 'random'
        config2 = create_string_config('test_attr', default='random', options=options)
        attr.validate(config2)
        
        # Test with 'none'
        config3 = create_string_config('test_attr', default='none', options=options)
        attr.validate(config3)
    
    def test_string_single_option(self):
        """
        Test behavior with only one option available.
        
        Edge case: with one option, should always return that option.
        """
        attr = StringAttribute('test_attr')
        options = ['only_option']
        config = create_string_config('test_attr', default='random', options=options)
        
        values = [attr.init_value(config) for _ in range(10)]
        
        # All should be the only option
        for v in values:
            self.assertEqual(v, 'only_option')
    
    def test_string_many_options(self):
        """
        Test behavior with many options available.
        
        Should be able to handle large option sets.
        """
        attr = StringAttribute('test_attr')
        options = [f'option_{i}' for i in range(50)]
        config = create_string_config('test_attr', default='random', options=options)
        
        values = [attr.init_value(config) for _ in range(200)]
        
        # Should use many different options
        unique_values = set(values)
        self.assertGreater(len(unique_values), 10, 
                         "Should use diverse options from large set")


class TestAttributeEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions across attribute types."""
    
    def test_float_attribute_with_equal_min_max(self):
        """
        Test FloatAttribute when min_value equals max_value.
        
        Edge case: should always return that single value.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', init_mean=5.0, init_stdev=2.0,
                                     mutate_rate=1.0, mutate_power=1.0,
                                     min_value=7.0, max_value=7.0)
        
        # All initialized and mutated values should be clamped to 7.0
        values = [attr.init_value(config) for _ in range(10)]
        for v in values:
            self.assertEqual(v, 7.0)
        
        # Mutations should also be clamped (any mutated value gets clamped to 7.0)
        mutated = attr.mutate_value(10.0, config)
        self.assertEqual(mutated, 7.0)
    
    def test_integer_attribute_with_single_value_range(self):
        """
        Test IntegerAttribute when min equals max.
        
        Edge case: should always return that single value.
        """
        attr = IntegerAttribute('test_attr')
        config = create_integer_config('test_attr', min_value=5, max_value=5)
        
        values = [attr.init_value(config) for _ in range(10)]
        for v in values:
            self.assertEqual(v, 5)
    
    def test_float_extreme_negative_values(self):
        """
        Test FloatAttribute with extreme negative values.
        
        Should handle large negative numbers correctly.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', init_mean=-1000.0, init_stdev=100.0,
                                     min_value=-2000.0, max_value=-500.0)
        
        values = [attr.init_value(config) for _ in range(100)]
        
        for v in values:
            self.assertGreaterEqual(v, -2000.0)
            self.assertLessEqual(v, -500.0)
    
    def test_float_extreme_positive_values(self):
        """
        Test FloatAttribute with extreme positive values.
        
        Should handle large positive numbers correctly.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', init_mean=1000.0, init_stdev=100.0,
                                     min_value=500.0, max_value=2000.0)
        
        values = [attr.init_value(config) for _ in range(100)]
        
        for v in values:
            self.assertGreaterEqual(v, 500.0)
            self.assertLessEqual(v, 2000.0)
    
    def test_combined_mutation_and_replacement_rates(self):
        """
        Test FloatAttribute with both mutation and replacement rates.
        
        Both operations should occur according to their rates.
        """
        attr = FloatAttribute('test_attr')
        config = create_float_config('test_attr', init_mean=0.0, init_stdev=1.0,
                                     mutate_rate=0.5, replace_rate=0.3,
                                     mutate_power=0.1, min_value=-100.0, max_value=100.0)
        
        initial_value = 50.0
        unchanged_count = 0
        
        # With total rate of 0.8, about 20% should be unchanged
        for _ in range(100):
            result = attr.mutate_value(initial_value, config)
            if result == initial_value:
                unchanged_count += 1
        
        # Should have some unchanged values
        self.assertGreater(unchanged_count, 5, 
                         "Some values should remain unchanged")
        self.assertLess(unchanged_count, 40, 
                       "Most values should change with combined rates")
    
    def test_config_item_name_generation(self):
        """
        Test that attribute generates correct config item names.
        
        Names should follow the pattern: {attr_name}_{item_name}
        """
        attr = FloatAttribute('weight')
        
        self.assertEqual(attr.config_item_name('init_mean'), 'weight_init_mean')
        self.assertEqual(attr.config_item_name('mutate_rate'), 'weight_mutate_rate')
        self.assertEqual(attr.config_item_name('max_value'), 'weight_max_value')


if __name__ == '__main__':
    unittest.main()
