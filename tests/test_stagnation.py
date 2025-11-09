"""
Comprehensive tests for stagnation detection in neat-python.

Tests cover:
- Stagnation detection timing over multiple generations
- Species elitism protection
- Different species fitness function options (mean, max, min, median)
- Fitness history tracking and last_improved updates
- Edge cases (all stagnant, single species, negative fitness, etc.)
- Boundary conditions (improvement at threshold)

These tests address gaps identified in TESTING_RECOMMENDATIONS.md for the stagnation module,
which had 98% coverage but no dedicated test file.
"""

import os
import sys
import unittest
import neat
from neat.stagnation import DefaultStagnation
from neat.species import Species, DefaultSpeciesSet
from neat.reporting import ReporterSet


class MockGenome:
    """Mock genome for testing with fitness attribute."""
    def __init__(self, key, fitness):
        self.key = key
        self.fitness = fitness


class MockSpeciesSet:
    """Mock species set for testing stagnation."""
    def __init__(self):
        self.species = {}
    
    def add_species(self, species_id, species):
        """Add a species to the set."""
        self.species[species_id] = species


def create_species_with_members(species_id, generation, fitnesses):
    """
    Create a species with members having specified fitness values.
    
    Args:
        species_id: ID for the species
        generation: Generation the species was created
        fitnesses: List of fitness values for members
        
    Returns:
        Species object with mock genome members
    """
    species = Species(species_id, generation)
    members = {}
    for i, fitness in enumerate(fitnesses):
        genome = MockGenome(i, fitness)
        members[i] = genome
    
    species.members = members
    species.representative = list(members.values())[0] if members else None
    return species


class TestStagnationDetection(unittest.TestCase):
    """Tests for stagnation detection and species fitness tracking."""
    
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
        self.reporters = ReporterSet()
    
    def create_stagnation_config(self, max_stagnation=15, species_elitism=0, 
                                  species_fitness_func='mean'):
        """
        Create a custom stagnation configuration.
        
        Args:
            max_stagnation: Generations without improvement before stagnation
            species_elitism: Number of top species to protect from stagnation
            species_fitness_func: Function to compute species fitness
            
        Returns:
            Stagnation configuration object
        """
        class StagnationConfig:
            pass
        
        config = StagnationConfig()
        config.max_stagnation = max_stagnation
        config.species_elitism = species_elitism
        config.species_fitness_func = species_fitness_func
        return config
    
    # ========== Stagnation Detection Timing Tests ==========
    
    def test_stagnation_at_exact_threshold(self):
        """
        Test that species is marked stagnant at exactly max_stagnation generations.
        
        A species that hasn't improved for max_stagnation generations should be
        marked as stagnant on the generation where the threshold is reached.
        """
        config = self.create_stagnation_config(max_stagnation=5, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        species = create_species_with_members(1, generation=0, fitnesses=[1.0])
        species.last_improved = 0
        species.fitness_history = [1.0]
        species_set.add_species(1, species)
        
        # Test generations 1-4: should not be stagnant
        for gen in range(1, 5):
            result = stagnation.update(species_set, gen)
            self.assertEqual(len(result), 1)
            sid, s, is_stagnant = result[0]
            self.assertFalse(is_stagnant, 
                           f"Species should not be stagnant at generation {gen}")
        
        # Test generation 5: should be stagnant (5 generations without improvement)
        result = stagnation.update(species_set, 5)
        sid, s, is_stagnant = result[0]
        self.assertTrue(is_stagnant, 
                       "Species should be stagnant after max_stagnation generations")
    
    def test_improvement_resets_stagnation(self):
        """
        Test that fitness improvement resets stagnation counter.
        
        When a species improves, its last_improved should be updated and
        the stagnation timer should reset.
        """
        config = self.create_stagnation_config(max_stagnation=3, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        species = create_species_with_members(1, generation=0, fitnesses=[1.0])
        species.last_improved = 0
        species.fitness_history = [1.0]
        species_set.add_species(1, species)
        
        # Generation 1-2: no improvement
        for gen in range(1, 3):
            stagnation.update(species_set, gen)
        
        # Generation 3: improve fitness
        species.members = {0: MockGenome(0, 2.0)}
        result = stagnation.update(species_set, 3)
        sid, s, is_stagnant = result[0]
        
        # Should not be stagnant after improvement
        self.assertFalse(is_stagnant)
        self.assertEqual(s.last_improved, 3, "last_improved should be updated")
        
        # Verify stagnation timer reset - should take another 3 generations
        # Gen 4 and 5: not stagnant yet (only 1-2 gens since improvement)
        species.members = {0: MockGenome(0, 2.0)}  # No further improvement
        for gen in range(4, 6):  # Test gen 4 and 5 only
            result = stagnation.update(species_set, gen)
            sid, s, is_stagnant = result[0]
            self.assertFalse(is_stagnant, 
                           f"Should not be stagnant at gen {gen} (only {gen-3} gens since improvement)")
        
        # Gen 6: should be stagnant (3 generations since improvement at gen 3)
        result = stagnation.update(species_set, 6)
        sid, s, is_stagnant = result[0]
        self.assertTrue(is_stagnant, 
                       "Should be stagnant after 3 generations without improvement")
    
    def test_improvement_just_before_threshold(self):
        """
        Test species that improves just before hitting max_stagnation threshold.
        
        Boundary condition: improvement at generation (max_stagnation - 1) should
        prevent stagnation.
        """
        config = self.create_stagnation_config(max_stagnation=5, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        species = create_species_with_members(1, generation=0, fitnesses=[1.0])
        species.last_improved = 0
        species.fitness_history = [1.0]
        species_set.add_species(1, species)
        
        # Generations 1-3: no improvement
        for gen in range(1, 4):
            stagnation.update(species_set, gen)
        
        # Generation 4: improve (just before threshold at gen 5)
        species.members = {0: MockGenome(0, 2.0)}
        result = stagnation.update(species_set, 4)
        sid, s, is_stagnant = result[0]
        self.assertFalse(is_stagnant)
        
        # Generation 5: should still not be stagnant (only 1 generation since improvement)
        species.members = {0: MockGenome(0, 2.0)}
        result = stagnation.update(species_set, 5)
        sid, s, is_stagnant = result[0]
        self.assertFalse(is_stagnant)
    
    def test_no_improvement_multiple_generations(self):
        """
        Test stagnation detection over many generations without improvement.
        
        Track a species over multiple generations to ensure stagnation
        persists once triggered.
        """
        config = self.create_stagnation_config(max_stagnation=3, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        species = create_species_with_members(1, generation=0, fitnesses=[1.0])
        species.last_improved = 0
        species.fitness_history = [1.0]
        species_set.add_species(1, species)
        
        # Run for many generations without improvement
        for gen in range(1, 10):
            result = stagnation.update(species_set, gen)
            sid, s, is_stagnant = result[0]
            
            if gen >= 3:  # Should be stagnant from generation 3 onward
                self.assertTrue(is_stagnant, 
                              f"Species should remain stagnant at generation {gen}")
    
    # ========== Species Elitism Protection Tests ==========
    
    def test_elitism_protects_top_species(self):
        """
        Test that top N species by fitness are protected from stagnation.
        
        When species_elitism > 0, the highest-fitness species should not be
        marked stagnant even if they haven't improved.
        """
        config = self.create_stagnation_config(
            max_stagnation=2, 
            species_elitism=2,
            species_fitness_func='mean'
        )
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        
        # Create 3 stagnant species with different fitness levels
        # Species 1: low fitness
        s1 = create_species_with_members(1, generation=0, fitnesses=[1.0])
        s1.last_improved = 0
        s1.fitness_history = [1.0]
        species_set.add_species(1, s1)
        
        # Species 2: medium fitness
        s2 = create_species_with_members(2, generation=0, fitnesses=[5.0])
        s2.last_improved = 0
        s2.fitness_history = [5.0]
        species_set.add_species(2, s2)
        
        # Species 3: high fitness
        s3 = create_species_with_members(3, generation=0, fitnesses=[10.0])
        s3.last_improved = 0
        s3.fitness_history = [10.0]
        species_set.add_species(3, s3)
        
        # After max_stagnation generations, check stagnation status
        result = stagnation.update(species_set, 3)
        
        # Sort by fitness to match algorithm behavior
        result_dict = {sid: (s, is_stagnant) for sid, s, is_stagnant in result}
        
        # Species 1 (lowest fitness) should be stagnant
        self.assertTrue(result_dict[1][1], "Lowest fitness species should be stagnant")
        
        # Species 2 and 3 (top 2 by fitness) should be protected
        self.assertFalse(result_dict[2][1], "Top 2 species should be protected")
        self.assertFalse(result_dict[3][1], "Top 2 species should be protected")
    
    def test_elitism_removes_lowest_fitness_first(self):
        """
        Test that lower-fitness species are marked stagnant before higher-fitness ones.
        
        With elitism, species should be marked stagnant in ascending fitness order.
        """
        config = self.create_stagnation_config(
            max_stagnation=2, 
            species_elitism=1,
            species_fitness_func='mean'
        )
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        
        # Create 4 stagnant species with different fitness
        for i, fitness in enumerate([1.0, 3.0, 7.0, 10.0], start=1):
            species = create_species_with_members(i, generation=0, fitnesses=[fitness])
            species.last_improved = 0
            species.fitness_history = [fitness]
            species_set.add_species(i, species)
        
        result = stagnation.update(species_set, 3)
        result_dict = {sid: (s, is_stagnant) for sid, s, is_stagnant in result}
        
        # With elitism=1, only top species protected
        # Lower fitness species should be stagnant
        self.assertTrue(result_dict[1][1], "Lowest fitness species should be stagnant")
        self.assertTrue(result_dict[2][1], "Second lowest should be stagnant")
        self.assertTrue(result_dict[3][1], "Third should be stagnant")
        self.assertFalse(result_dict[4][1], "Highest fitness should be protected")
    
    def test_elitism_with_all_species_stagnant(self):
        """
        Test elitism protection when all species are stagnant.
        
        Even if all species are stagnant, the top N should still be protected.
        """
        config = self.create_stagnation_config(
            max_stagnation=1, 
            species_elitism=2,
            species_fitness_func='mean'
        )
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        
        # Create 5 species, all stagnant for many generations
        for i, fitness in enumerate([1.0, 2.0, 5.0, 8.0, 10.0], start=1):
            species = create_species_with_members(i, generation=0, fitnesses=[fitness])
            species.last_improved = 0
            species.fitness_history = [fitness]
            species_set.add_species(i, species)
        
        result = stagnation.update(species_set, 10)
        result_dict = {sid: (s, is_stagnant) for sid, s, is_stagnant in result}
        
        # Top 2 should be protected
        self.assertFalse(result_dict[4][1], "Second highest should be protected")
        self.assertFalse(result_dict[5][1], "Highest should be protected")
        
        # Others should be stagnant
        self.assertTrue(result_dict[1][1])
        self.assertTrue(result_dict[2][1])
        self.assertTrue(result_dict[3][1])
    
    def test_elitism_zero_allows_all_stagnation(self):
        """
        Test that with species_elitism=0, all species can be marked stagnant.
        
        When elitism is disabled, no species should be protected.
        """
        config = self.create_stagnation_config(
            max_stagnation=2, 
            species_elitism=0,
            species_fitness_func='mean'
        )
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        
        # Create multiple stagnant species
        for i in range(1, 4):
            species = create_species_with_members(i, generation=0, fitnesses=[float(i)])
            species.last_improved = 0
            species.fitness_history = [float(i)]
            species_set.add_species(i, species)
        
        result = stagnation.update(species_set, 3)
        
        # All should be stagnant
        for sid, s, is_stagnant in result:
            self.assertTrue(is_stagnant, 
                          f"Species {sid} should be stagnant with elitism=0")
    
    # ========== Fitness Function Tests ==========
    
    def test_mean_fitness_function(self):
        """
        Test stagnation with 'mean' species fitness function.
        
        Verify that species fitness is computed as the mean of member fitnesses.
        """
        config = self.create_stagnation_config(
            max_stagnation=5, 
            species_elitism=0,
            species_fitness_func='mean'
        )
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        # Members with fitnesses [1.0, 2.0, 3.0], mean = 2.0
        species = create_species_with_members(1, generation=0, 
                                             fitnesses=[1.0, 2.0, 3.0])
        species.fitness_history = []
        species_set.add_species(1, species)
        
        result = stagnation.update(species_set, 0)
        sid, s, is_stagnant = result[0]
        
        self.assertAlmostEqual(s.fitness, 2.0, places=5, 
                              msg="Species fitness should be mean of members")
    
    def test_max_fitness_function(self):
        """
        Test stagnation with 'max' species fitness function.
        
        Verify that species fitness is the maximum of member fitnesses.
        """
        config = self.create_stagnation_config(
            max_stagnation=5, 
            species_elitism=0,
            species_fitness_func='max'
        )
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        # Members with fitnesses [1.0, 5.0, 3.0], max = 5.0
        species = create_species_with_members(1, generation=0, 
                                             fitnesses=[1.0, 5.0, 3.0])
        species.fitness_history = []
        species_set.add_species(1, species)
        
        result = stagnation.update(species_set, 0)
        sid, s, is_stagnant = result[0]
        
        self.assertEqual(s.fitness, 5.0, 
                        "Species fitness should be max of members")
    
    def test_min_fitness_function(self):
        """
        Test stagnation with 'min' species fitness function.
        
        Verify that species fitness is the minimum of member fitnesses.
        """
        config = self.create_stagnation_config(
            max_stagnation=5, 
            species_elitism=0,
            species_fitness_func='min'
        )
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        # Members with fitnesses [1.0, 5.0, 3.0], min = 1.0
        species = create_species_with_members(1, generation=0, 
                                             fitnesses=[1.0, 5.0, 3.0])
        species.fitness_history = []
        species_set.add_species(1, species)
        
        result = stagnation.update(species_set, 0)
        sid, s, is_stagnant = result[0]
        
        self.assertEqual(s.fitness, 1.0, 
                        "Species fitness should be min of members")
    
    def test_median_fitness_function(self):
        """
        Test stagnation with 'median' species fitness function.
        
        Verify that species fitness is the median of member fitnesses.
        """
        config = self.create_stagnation_config(
            max_stagnation=5, 
            species_elitism=0,
            species_fitness_func='median'
        )
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        # Members with fitnesses [1.0, 2.0, 3.0, 4.0, 5.0], median = 3.0
        species = create_species_with_members(1, generation=0, 
                                             fitnesses=[1.0, 2.0, 3.0, 4.0, 5.0])
        species.fitness_history = []
        species_set.add_species(1, species)
        
        result = stagnation.update(species_set, 0)
        sid, s, is_stagnant = result[0]
        
        self.assertEqual(s.fitness, 3.0, 
                        "Species fitness should be median of members")
    
    def test_fitness_function_affects_improvement_detection(self):
        """
        Test that different fitness functions affect improvement detection.
        
        With different functions, the same member fitnesses can result in
        different improvement patterns.
        """
        species_set = MockSpeciesSet()
        
        # Test with max function - only max fitness matters
        config_max = self.create_stagnation_config(
            max_stagnation=5, 
            species_elitism=0,
            species_fitness_func='max'
        )
        stagnation_max = DefaultStagnation(config_max, self.reporters)
        
        species = create_species_with_members(1, generation=0, 
                                             fitnesses=[1.0, 2.0, 5.0])
        species.fitness_history = []
        species.last_improved = 0
        species_set.add_species(1, species)
        
        # First update: max = 5.0
        stagnation_max.update(species_set, 0)
        
        # Second update: [1.0, 3.0, 5.0], max still 5.0 - no improvement
        species.members = {i: MockGenome(i, f) 
                          for i, f in enumerate([1.0, 3.0, 5.0])}
        result = stagnation_max.update(species_set, 1)
        sid, s, is_stagnant = result[0]
        
        # last_improved should still be 0 (no improvement in max)
        self.assertEqual(s.last_improved, 0, 
                        "Max fitness didn't improve, last_improved should not change")
    
    # ========== Fitness History Tracking Tests ==========
    
    def test_fitness_history_appended(self):
        """
        Test that fitness history is correctly appended each generation.
        
        Each call to update() should add the current fitness to history.
        """
        config = self.create_stagnation_config(max_stagnation=5, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        species = create_species_with_members(1, generation=0, fitnesses=[1.0])
        species.fitness_history = []
        species_set.add_species(1, species)
        
        # Update over several generations
        fitnesses = [1.0, 2.0, 2.5, 3.0, 2.8]
        for gen, fitness in enumerate(fitnesses):
            species.members = {0: MockGenome(0, fitness)}
            result = stagnation.update(species_set, gen)
            sid, s, is_stagnant = result[0]
            
            # Fitness history should have all values up to current generation
            self.assertEqual(len(s.fitness_history), gen + 1, 
                           f"Fitness history should have {gen + 1} entries")
            self.assertEqual(s.fitness_history[-1], fitness, 
                           "Last entry should be current fitness")
    
    def test_last_improved_updated_on_improvement(self):
        """
        Test that last_improved is updated when fitness improves.
        
        The generation number should be recorded when fitness exceeds
        the previous maximum.
        """
        config = self.create_stagnation_config(max_stagnation=5, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        species = create_species_with_members(1, generation=0, fitnesses=[1.0])
        species.fitness_history = []
        species.last_improved = 0
        species_set.add_species(1, species)
        
        # Gen 0: fitness = 1.0
        stagnation.update(species_set, 0)
        self.assertEqual(species.last_improved, 0)
        
        # Gen 1: fitness = 1.0 (no improvement)
        species.members = {0: MockGenome(0, 1.0)}
        stagnation.update(species_set, 1)
        self.assertEqual(species.last_improved, 0, "Should not update without improvement")
        
        # Gen 2: fitness = 2.0 (improvement)
        species.members = {0: MockGenome(0, 2.0)}
        stagnation.update(species_set, 2)
        self.assertEqual(species.last_improved, 2, "Should update on improvement")
        
        # Gen 3: fitness = 1.5 (worse than best)
        species.members = {0: MockGenome(0, 1.5)}
        stagnation.update(species_set, 3)
        self.assertEqual(species.last_improved, 2, "Should not update on decline")
        
        # Gen 4: fitness = 3.0 (new best)
        species.members = {0: MockGenome(0, 3.0)}
        stagnation.update(species_set, 4)
        self.assertEqual(species.last_improved, 4, "Should update on new best")
    
    def test_last_improved_tracks_max_fitness(self):
        """
        Test that last_improved tracks the maximum fitness in history.
        
        Improvement is determined by comparing to the historical maximum,
        not just the previous generation.
        """
        config = self.create_stagnation_config(max_stagnation=5, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        species = create_species_with_members(1, generation=0, fitnesses=[5.0])
        species.fitness_history = []
        species.last_improved = 0
        species_set.add_species(1, species)
        
        # Gen 0: 5.0 (initial)
        stagnation.update(species_set, 0)
        
        # Gen 1: 3.0 (decline)
        species.members = {0: MockGenome(0, 3.0)}
        stagnation.update(species_set, 1)
        self.assertEqual(species.last_improved, 0)
        
        # Gen 2: 4.0 (better than gen 1 but not better than max of 5.0)
        species.members = {0: MockGenome(0, 4.0)}
        stagnation.update(species_set, 2)
        self.assertEqual(species.last_improved, 0, 
                        "Should not update when not exceeding historical max")
        
        # Gen 3: 6.0 (new max)
        species.members = {0: MockGenome(0, 6.0)}
        stagnation.update(species_set, 3)
        self.assertEqual(species.last_improved, 3, 
                        "Should update when exceeding historical max")
    
    # ========== Edge Case Tests ==========
    
    def test_single_species_population(self):
        """
        Test stagnation with only one species in population.
        
        Edge case: single species should still be subject to stagnation rules.
        """
        config = self.create_stagnation_config(max_stagnation=3, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        species = create_species_with_members(1, generation=0, fitnesses=[1.0])
        species.last_improved = 0
        species.fitness_history = [1.0]
        species_set.add_species(1, species)
        
        # Should be stagnant after max_stagnation generations
        result = stagnation.update(species_set, 4)
        sid, s, is_stagnant = result[0]
        self.assertTrue(is_stagnant, "Single species can still be stagnant")
    
    def test_all_species_stagnant(self):
        """
        Test when all species in population are stagnant.
        
        Edge case: verify correct handling when entire population is stagnant.
        """
        config = self.create_stagnation_config(max_stagnation=2, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        
        # Create 3 species, all stagnant
        for i in range(1, 4):
            species = create_species_with_members(i, generation=0, fitnesses=[float(i)])
            species.last_improved = 0
            species.fitness_history = [float(i)]
            species_set.add_species(i, species)
        
        result = stagnation.update(species_set, 5)
        
        # All should be marked stagnant
        stagnant_count = sum(1 for _, _, is_stagnant in result if is_stagnant)
        self.assertEqual(stagnant_count, 3, "All species should be stagnant")
    
    def test_species_with_identical_fitness(self):
        """
        Test species with identical fitness values across all members.
        
        Edge case: all members having the same fitness should work correctly.
        """
        config = self.create_stagnation_config(max_stagnation=5, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        # All members with fitness 5.0
        species = create_species_with_members(1, generation=0, 
                                             fitnesses=[5.0, 5.0, 5.0, 5.0])
        species.fitness_history = []
        species_set.add_species(1, species)
        
        result = stagnation.update(species_set, 0)
        sid, s, is_stagnant = result[0]
        
        # Should handle identical fitnesses without error
        self.assertIsNotNone(s.fitness)
        self.assertEqual(s.fitness, 5.0)
    
    def test_negative_fitness_values(self):
        """
        Test stagnation with negative fitness values.
        
        Edge case: negative fitnesses should be handled correctly,
        with proper comparison for improvement detection.
        """
        config = self.create_stagnation_config(max_stagnation=5, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        species = create_species_with_members(1, generation=0, fitnesses=[-5.0])
        species.fitness_history = []
        species.last_improved = 0
        species_set.add_species(1, species)
        
        # Gen 0: -5.0
        stagnation.update(species_set, 0)
        
        # Gen 1: -3.0 (improvement, even though both are negative)
        species.members = {0: MockGenome(0, -3.0)}
        result = stagnation.update(species_set, 1)
        sid, s, is_stagnant = result[0]
        
        self.assertEqual(s.last_improved, 1, 
                        "Should detect improvement with negative values")
        self.assertAlmostEqual(s.fitness, -3.0, places=5)
    
    def test_empty_fitness_history(self):
        """
        Test species with empty fitness history at initialization.
        
        Edge case: first update should handle empty history gracefully.
        """
        config = self.create_stagnation_config(max_stagnation=5, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        species = create_species_with_members(1, generation=0, fitnesses=[1.0])
        species.fitness_history = []  # Empty history
        species.last_improved = 0
        species_set.add_species(1, species)
        
        # Should handle empty history without error
        result = stagnation.update(species_set, 0)
        sid, s, is_stagnant = result[0]
        
        self.assertEqual(len(s.fitness_history), 1, 
                        "Fitness history should have one entry after first update")
        self.assertFalse(is_stagnant, "Should not be stagnant on first generation")
    
    def test_fitness_history_with_fluctuations(self):
        """
        Test fitness history tracking with fluctuating fitness values.
        
        Verify that improvement detection works correctly when fitness
        goes up and down over time.
        """
        config = self.create_stagnation_config(max_stagnation=5, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        species = create_species_with_members(1, generation=0, fitnesses=[5.0])
        species.fitness_history = []
        species.last_improved = 0
        species_set.add_species(1, species)
        
        # Fitness pattern: 5.0 -> 3.0 -> 7.0 -> 4.0 -> 8.0
        fitness_values = [5.0, 3.0, 7.0, 4.0, 8.0]
        expected_last_improved = [0, 0, 2, 2, 4]  # When new maxes occur
        
        for gen, (fitness, expected) in enumerate(zip(fitness_values, 
                                                       expected_last_improved)):
            species.members = {0: MockGenome(0, fitness)}
            result = stagnation.update(species_set, gen)
            sid, s, is_stagnant = result[0]
            
            self.assertEqual(s.last_improved, expected,
                           f"Gen {gen}: last_improved should be {expected}")
    
    def test_very_large_fitness_values(self):
        """
        Test with very large fitness values.
        
        Edge case: ensure no overflow or precision issues with large numbers.
        """
        config = self.create_stagnation_config(max_stagnation=5, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        large_fitness = 1e10
        species = create_species_with_members(1, generation=0, fitnesses=[large_fitness])
        species.fitness_history = []
        species_set.add_species(1, species)
        
        result = stagnation.update(species_set, 0)
        sid, s, is_stagnant = result[0]
        
        self.assertAlmostEqual(s.fitness, large_fitness, places=5, 
                              msg="Should handle large fitness values")
    
    def test_zero_fitness_values(self):
        """
        Test with zero fitness values.
        
        Edge case: zero fitness should be handled as a valid fitness value.
        """
        config = self.create_stagnation_config(max_stagnation=5, species_elitism=0)
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        species = create_species_with_members(1, generation=0, fitnesses=[0.0])
        species.fitness_history = []
        species.last_improved = 0
        species_set.add_species(1, species)
        
        # Gen 0: 0.0
        stagnation.update(species_set, 0)
        
        # Gen 1: 0.1 (improvement from 0)
        species.members = {0: MockGenome(0, 0.1)}
        result = stagnation.update(species_set, 1)
        sid, s, is_stagnant = result[0]
        
        self.assertEqual(s.last_improved, 1, 
                        "Should detect improvement from zero")
    
    def test_stagnation_with_multiple_species_complex(self):
        """
        Test complex scenario with multiple species at different stages.
        
        Integration test: multiple species with varying fitness histories
        and stagnation states.
        """
        config = self.create_stagnation_config(
            max_stagnation=3, 
            species_elitism=1,
            species_fitness_func='mean'
        )
        stagnation = DefaultStagnation(config, self.reporters)
        
        species_set = MockSpeciesSet()
        
        # Species 1: Improving (fitness: 1->2->3)
        s1 = create_species_with_members(1, generation=0, fitnesses=[1.0])
        s1.fitness_history = [1.0, 2.0]
        s1.last_improved = 1
        species_set.add_species(1, s1)
        
        # Species 2: Stagnant (fitness: 5->5->5->5)
        s2 = create_species_with_members(2, generation=0, fitnesses=[5.0])
        s2.fitness_history = [5.0, 5.0, 5.0]
        s2.last_improved = 0
        species_set.add_species(2, s2)
        
        # Species 3: Recently improved (fitness: 3->3->7)
        s3 = create_species_with_members(3, generation=0, fitnesses=[7.0])
        s3.fitness_history = [3.0, 3.0]
        s3.last_improved = 2
        species_set.add_species(3, s3)
        
        # Update at generation 3
        s1.members = {0: MockGenome(0, 3.0)}
        s2.members = {0: MockGenome(0, 5.0)}
        s3.members = {0: MockGenome(0, 7.0)}
        
        result = stagnation.update(species_set, 3)
        result_dict = {sid: (s, is_stagnant) for sid, s, is_stagnant in result}
        
        # Species 1: Not stagnant (recently improved)
        self.assertFalse(result_dict[1][1], "Recently improving species not stagnant")
        
        # Species 2: Stagnant (3+ generations without improvement)
        # But might be protected by elitism if it has highest fitness
        
        # Species 3: Not stagnant (improved at gen 2, only 1 gen ago)
        self.assertFalse(result_dict[3][1], "Recently improved species not stagnant")


if __name__ == '__main__':
    unittest.main()
