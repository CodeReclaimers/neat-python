"""
Tests for statistics collection and reporting functionality.
"""
import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock
import time

import neat
from neat.reporting import ReporterSet, BaseReporter, StdOutReporter
from neat.statistics import StatisticsReporter
from neat.genome import DefaultGenome
from neat.species import Species


class MockGenome:
    """Mock genome for testing."""
    def __init__(self, key, fitness=0.0, num_nodes=5, num_connections=10):
        self.key = key
        self.fitness = fitness
        self._num_nodes = num_nodes
        self._num_connections = num_connections
    
    def size(self):
        return (self._num_nodes, self._num_connections)


class MockSpecies:
    """Mock species for testing."""
    def __init__(self, sid, members, fitness=None, adjusted_fitness=None, created=0, last_improved=0):
        self.key = sid
        self.members = {g.key: g for g in members}
        self.fitness = fitness
        self.adjusted_fitness = adjusted_fitness
        self.created = created
        self.last_improved = last_improved


class MockSpeciesSet:
    """Mock species set for testing."""
    def __init__(self, species_dict):
        self.species = species_dict
    
    def get_species_id(self, genome_key):
        for sid, species in self.species.items():
            if genome_key in species.members:
                return sid
        return None


class TestReporterSet(unittest.TestCase):
    """Tests for ReporterSet class."""
    
    def setUp(self):
        self.reporter_set = ReporterSet()
    
    def test_add_reporter(self):
        """Test adding a reporter."""
        reporter = BaseReporter()
        self.reporter_set.add(reporter)
        self.assertIn(reporter, self.reporter_set.reporters)
    
    def test_add_multiple_reporters(self):
        """Test adding multiple reporters."""
        r1 = BaseReporter()
        r2 = BaseReporter()
        self.reporter_set.add(r1)
        self.reporter_set.add(r2)
        self.assertEqual(len(self.reporter_set.reporters), 2)
        self.assertIn(r1, self.reporter_set.reporters)
        self.assertIn(r2, self.reporter_set.reporters)
    
    def test_remove_reporter(self):
        """Test removing a reporter."""
        reporter = BaseReporter()
        self.reporter_set.add(reporter)
        self.reporter_set.remove(reporter)
        self.assertNotIn(reporter, self.reporter_set.reporters)
    
    def test_start_generation_dispatches_to_all(self):
        """Test that start_generation is called on all reporters."""
        r1 = MagicMock(spec=BaseReporter)
        r2 = MagicMock(spec=BaseReporter)
        self.reporter_set.add(r1)
        self.reporter_set.add(r2)
        
        self.reporter_set.start_generation(5)
        
        r1.start_generation.assert_called_once_with(5)
        r2.start_generation.assert_called_once_with(5)
    
    def test_end_generation_dispatches_to_all(self):
        """Test that end_generation is called on all reporters."""
        r1 = MagicMock(spec=BaseReporter)
        r2 = MagicMock(spec=BaseReporter)
        self.reporter_set.add(r1)
        self.reporter_set.add(r2)
        
        config = MagicMock()
        population = {}
        species_set = MagicMock()
        
        self.reporter_set.end_generation(config, population, species_set)
        
        r1.end_generation.assert_called_once_with(config, population, species_set)
        r2.end_generation.assert_called_once_with(config, population, species_set)
    
    def test_post_evaluate_dispatches_to_all(self):
        """Test that post_evaluate is called on all reporters."""
        r1 = MagicMock(spec=BaseReporter)
        r2 = MagicMock(spec=BaseReporter)
        self.reporter_set.add(r1)
        self.reporter_set.add(r2)
        
        config = MagicMock()
        population = {}
        species = MagicMock()
        best_genome = MockGenome(1, 10.0)
        
        self.reporter_set.post_evaluate(config, population, species, best_genome)
        
        r1.post_evaluate.assert_called_once_with(config, population, species, best_genome)
        r2.post_evaluate.assert_called_once_with(config, population, species, best_genome)
    
    def test_post_reproduction_dispatches_to_all(self):
        """Test that post_reproduction is called on all reporters."""
        r1 = MagicMock(spec=BaseReporter)
        r2 = MagicMock(spec=BaseReporter)
        self.reporter_set.add(r1)
        self.reporter_set.add(r2)
        
        config = MagicMock()
        population = {}
        species = MagicMock()
        
        self.reporter_set.post_reproduction(config, population, species)
        
        r1.post_reproduction.assert_called_once_with(config, population, species)
        r2.post_reproduction.assert_called_once_with(config, population, species)
    
    def test_complete_extinction_dispatches_to_all(self):
        """Test that complete_extinction is called on all reporters."""
        r1 = MagicMock(spec=BaseReporter)
        r2 = MagicMock(spec=BaseReporter)
        self.reporter_set.add(r1)
        self.reporter_set.add(r2)
        
        self.reporter_set.complete_extinction()
        
        r1.complete_extinction.assert_called_once()
        r2.complete_extinction.assert_called_once()
    
    def test_found_solution_dispatches_to_all(self):
        """Test that found_solution is called on all reporters."""
        r1 = MagicMock(spec=BaseReporter)
        r2 = MagicMock(spec=BaseReporter)
        self.reporter_set.add(r1)
        self.reporter_set.add(r2)
        
        config = MagicMock()
        best = MockGenome(1, 10.0)
        
        self.reporter_set.found_solution(config, 5, best)
        
        r1.found_solution.assert_called_once_with(config, 5, best)
        r2.found_solution.assert_called_once_with(config, 5, best)
    
    def test_species_stagnant_dispatches_to_all(self):
        """Test that species_stagnant is called on all reporters."""
        r1 = MagicMock(spec=BaseReporter)
        r2 = MagicMock(spec=BaseReporter)
        self.reporter_set.add(r1)
        self.reporter_set.add(r2)
        
        species = MagicMock()
        
        self.reporter_set.species_stagnant(1, species)
        
        r1.species_stagnant.assert_called_once_with(1, species)
        r2.species_stagnant.assert_called_once_with(1, species)
    
    def test_info_dispatches_to_all(self):
        """Test that info is called on all reporters."""
        r1 = MagicMock(spec=BaseReporter)
        r2 = MagicMock(spec=BaseReporter)
        self.reporter_set.add(r1)
        self.reporter_set.add(r2)
        
        self.reporter_set.info("Test message")
        
        r1.info.assert_called_once_with("Test message")
        r2.info.assert_called_once_with("Test message")


class TestStdOutReporter(unittest.TestCase):
    """Tests for StdOutReporter class."""
    
    def test_initialization_with_detail(self):
        """Test reporter initialization with species detail enabled."""
        reporter = StdOutReporter(show_species_detail=True)
        self.assertTrue(reporter.show_species_detail)
        self.assertIsNone(reporter.generation)
        self.assertIsNone(reporter.generation_start_time)
        self.assertEqual(reporter.generation_times, [])
        self.assertEqual(reporter.num_extinctions, 0)
    
    def test_initialization_without_detail(self):
        """Test reporter initialization with species detail disabled."""
        reporter = StdOutReporter(show_species_detail=False)
        self.assertFalse(reporter.show_species_detail)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_start_generation_output(self, mock_stdout):
        """Test start_generation produces correct output."""
        reporter = StdOutReporter(show_species_detail=False)
        reporter.start_generation(5)
        
        output = mock_stdout.getvalue()
        self.assertIn("Running generation 5", output)
        self.assertEqual(reporter.generation, 5)
        self.assertIsNotNone(reporter.generation_start_time)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_end_generation_basic_output(self, mock_stdout):
        """Test end_generation with basic output (no species detail)."""
        reporter = StdOutReporter(show_species_detail=False)
        reporter.start_generation(1)
        
        population = {i: MockGenome(i) for i in range(10)}
        species_set = MockSpeciesSet({1: MockSpecies(1, list(population.values())[:5]),
                                      2: MockSpecies(2, list(population.values())[5:])})
        
        config = MagicMock()
        reporter.end_generation(config, population, species_set)
        
        output = mock_stdout.getvalue()
        self.assertIn("Population of 10 members in 2 species", output)
        self.assertIn("Total extinctions: 0", output)
        self.assertIn("Generation time:", output)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_end_generation_detailed_output(self, mock_stdout):
        """Test end_generation with species detail enabled."""
        reporter = StdOutReporter(show_species_detail=True)
        reporter.generation = 10
        reporter.generation_start_time = time.time()
        
        members1 = [MockGenome(i, fitness=float(i)) for i in range(5)]
        members2 = [MockGenome(i, fitness=float(i)) for i in range(5, 10)]
        
        species1 = MockSpecies(1, members1, fitness=2.5, adjusted_fitness=1.2, created=5, last_improved=8)
        species2 = MockSpecies(2, members2, fitness=7.5, adjusted_fitness=3.7, created=7, last_improved=10)
        
        population = {g.key: g for g in members1 + members2}
        species_set = MockSpeciesSet({1: species1, 2: species2})
        
        config = MagicMock()
        reporter.end_generation(config, population, species_set)
        
        output = mock_stdout.getvalue()
        self.assertIn("ID   age  size   fitness   adj fit  stag", output)
        self.assertIn("Population of 10 members in 2 species", output)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_end_generation_tracks_timing(self, mock_stdout):
        """Test that end_generation tracks timing across multiple calls."""
        reporter = StdOutReporter(show_species_detail=False)
        
        population = {i: MockGenome(i) for i in range(5)}
        species_set = MockSpeciesSet({1: MockSpecies(1, list(population.values()))})
        config = MagicMock()
        
        # First generation
        reporter.start_generation(1)
        time.sleep(0.01)  # Small delay
        reporter.end_generation(config, population, species_set)
        
        self.assertEqual(len(reporter.generation_times), 1)
        
        # Second generation
        reporter.start_generation(2)
        time.sleep(0.01)
        reporter.end_generation(config, population, species_set)
        
        output = mock_stdout.getvalue()
        self.assertEqual(len(reporter.generation_times), 2)
        self.assertIn("average", output)  # Average should appear after 2+ generations
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_end_generation_limits_timing_history(self, mock_stdout):
        """Test that timing history is limited to last 10 generations."""
        reporter = StdOutReporter(show_species_detail=False)
        
        population = {i: MockGenome(i) for i in range(5)}
        species_set = MockSpeciesSet({1: MockSpecies(1, list(population.values()))})
        config = MagicMock()
        
        # Run 15 generations
        for gen in range(15):
            reporter.start_generation(gen)
            reporter.end_generation(config, population, species_set)
        
        # Should only keep last 10
        self.assertEqual(len(reporter.generation_times), 10)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_post_evaluate_output(self, mock_stdout):
        """Test post_evaluate produces correct output."""
        reporter = StdOutReporter(show_species_detail=False)
        
        population = {i: MockGenome(i, fitness=float(i)) for i in range(10)}
        best_genome = MockGenome(15, fitness=20.0, num_nodes=8, num_connections=15)
        species_set = MockSpeciesSet({1: MockSpecies(1, list(population.values())[:5]),
                                      2: MockSpecies(2, list(population.values())[5:] + [best_genome])})
        
        config = MagicMock()
        reporter.post_evaluate(config, population, species_set, best_genome)
        
        output = mock_stdout.getvalue()
        self.assertIn("average fitness:", output)
        self.assertIn("stdev:", output)
        self.assertIn("Best fitness: 20.00000", output)
        self.assertIn("species 2", output)
        self.assertIn("id 15", output)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_complete_extinction_output(self, mock_stdout):
        """Test complete_extinction increments counter and outputs message."""
        reporter = StdOutReporter(show_species_detail=False)
        
        self.assertEqual(reporter.num_extinctions, 0)
        reporter.complete_extinction()
        self.assertEqual(reporter.num_extinctions, 1)
        
        output = mock_stdout.getvalue()
        self.assertIn("All species extinct", output)
        
        reporter.complete_extinction()
        self.assertEqual(reporter.num_extinctions, 2)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_found_solution_output(self, mock_stdout):
        """Test found_solution outputs correct message."""
        reporter = StdOutReporter(show_species_detail=False)
        reporter.generation = 42
        
        best = MockGenome(1, fitness=100.0, num_nodes=10, num_connections=20)
        config = MagicMock()
        
        reporter.found_solution(config, 42, best)
        
        output = mock_stdout.getvalue()
        self.assertIn("generation 42", output)
        self.assertIn("meets fitness threshold", output)
        self.assertIn("complexity:", output)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_species_stagnant_with_detail(self, mock_stdout):
        """Test species_stagnant with detail enabled."""
        reporter = StdOutReporter(show_species_detail=True)
        
        members = [MockGenome(i) for i in range(5)]
        species = MockSpecies(3, members)
        
        reporter.species_stagnant(3, species)
        
        output = mock_stdout.getvalue()
        self.assertIn("Species 3", output)
        self.assertIn("5 members", output)
        self.assertIn("stagnated", output)
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_species_stagnant_without_detail(self, mock_stdout):
        """Test species_stagnant with detail disabled produces no output."""
        reporter = StdOutReporter(show_species_detail=False)
        
        members = [MockGenome(i) for i in range(5)]
        species = MockSpecies(3, members)
        
        reporter.species_stagnant(3, species)
        
        output = mock_stdout.getvalue()
        self.assertEqual(output, "")
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_info_output(self, mock_stdout):
        """Test info method outputs message."""
        reporter = StdOutReporter(show_species_detail=False)
        
        reporter.info("Test information message")
        
        output = mock_stdout.getvalue()
        self.assertIn("Test information message", output)


class TestStatisticsReporter(unittest.TestCase):
    """Tests for StatisticsReporter class."""
    
    def setUp(self):
        self.stats = StatisticsReporter()
    
    def test_initialization(self):
        """Test statistics reporter initialization."""
        self.assertEqual(self.stats.most_fit_genomes, [])
        self.assertEqual(self.stats.generation_statistics, [])
    
    def test_post_evaluate_stores_best_genome(self):
        """Test that post_evaluate stores the best genome."""
        best_genome = MockGenome(1, fitness=10.0)
        population = {1: best_genome}
        species_set = MockSpeciesSet({1: MockSpecies(1, [best_genome])})
        config = MagicMock()
        
        self.stats.post_evaluate(config, population, species_set, best_genome)
        
        self.assertEqual(len(self.stats.most_fit_genomes), 1)
        self.assertEqual(self.stats.most_fit_genomes[0].fitness, 10.0)
    
    def test_post_evaluate_deep_copies_genome(self):
        """Test that post_evaluate deep copies genomes."""
        best_genome = MockGenome(1, fitness=10.0)
        population = {1: best_genome}
        species_set = MockSpeciesSet({1: MockSpecies(1, [best_genome])})
        config = MagicMock()
        
        self.stats.post_evaluate(config, population, species_set, best_genome)
        
        # Modify original
        best_genome.fitness = 5.0
        
        # Stored copy should be unchanged
        self.assertEqual(self.stats.most_fit_genomes[0].fitness, 10.0)
    
    def test_post_evaluate_stores_species_stats(self):
        """Test that post_evaluate stores species statistics."""
        g1 = MockGenome(1, fitness=5.0)
        g2 = MockGenome(2, fitness=7.0)
        g3 = MockGenome(3, fitness=3.0)
        
        species1 = MockSpecies(1, [g1, g2])
        species2 = MockSpecies(2, [g3])
        
        population = {1: g1, 2: g2, 3: g3}
        species_set = MockSpeciesSet({1: species1, 2: species2})
        config = MagicMock()
        
        self.stats.post_evaluate(config, population, species_set, g2)
        
        self.assertEqual(len(self.stats.generation_statistics), 1)
        gen_stats = self.stats.generation_statistics[0]
        self.assertEqual(gen_stats[1][1], 5.0)
        self.assertEqual(gen_stats[1][2], 7.0)
        self.assertEqual(gen_stats[2][3], 3.0)
    
    def test_post_evaluate_multiple_generations(self):
        """Test collecting statistics across multiple generations."""
        config = MagicMock()
        
        # Generation 1
        g1 = MockGenome(1, fitness=5.0)
        population1 = {1: g1}
        species_set1 = MockSpeciesSet({1: MockSpecies(1, [g1])})
        self.stats.post_evaluate(config, population1, species_set1, g1)
        
        # Generation 2
        g2 = MockGenome(2, fitness=10.0)
        population2 = {2: g2}
        species_set2 = MockSpeciesSet({1: MockSpecies(1, [g2])})
        self.stats.post_evaluate(config, population2, species_set2, g2)
        
        self.assertEqual(len(self.stats.most_fit_genomes), 2)
        self.assertEqual(len(self.stats.generation_statistics), 2)
    
    def test_get_fitness_mean(self):
        """Test calculating mean fitness across generations."""
        config = MagicMock()
        
        # Generation 1: fitness values [1.0, 2.0, 3.0] -> mean = 2.0
        g1 = MockGenome(1, fitness=1.0)
        g2 = MockGenome(2, fitness=2.0)
        g3 = MockGenome(3, fitness=3.0)
        species1 = MockSpecies(1, [g1, g2, g3])
        population1 = {1: g1, 2: g2, 3: g3}
        species_set1 = MockSpeciesSet({1: species1})
        self.stats.post_evaluate(config, population1, species_set1, g3)
        
        # Generation 2: fitness values [4.0, 5.0, 6.0] -> mean = 5.0
        g4 = MockGenome(4, fitness=4.0)
        g5 = MockGenome(5, fitness=5.0)
        g6 = MockGenome(6, fitness=6.0)
        species2 = MockSpecies(1, [g4, g5, g6])
        population2 = {4: g4, 5: g5, 6: g6}
        species_set2 = MockSpeciesSet({1: species2})
        self.stats.post_evaluate(config, population2, species_set2, g6)
        
        means = self.stats.get_fitness_mean()
        self.assertEqual(len(means), 2)
        self.assertAlmostEqual(means[0], 2.0)
        self.assertAlmostEqual(means[1], 5.0)
    
    def test_get_fitness_stdev(self):
        """Test calculating fitness standard deviation across generations."""
        config = MagicMock()
        
        # Generation with varying fitness
        g1 = MockGenome(1, fitness=1.0)
        g2 = MockGenome(2, fitness=5.0)
        g3 = MockGenome(3, fitness=9.0)
        species1 = MockSpecies(1, [g1, g2, g3])
        population1 = {1: g1, 2: g2, 3: g3}
        species_set1 = MockSpeciesSet({1: species1})
        self.stats.post_evaluate(config, population1, species_set1, g3)
        
        stdevs = self.stats.get_fitness_stdev()
        self.assertEqual(len(stdevs), 1)
        self.assertGreater(stdevs[0], 0)  # Should have non-zero stdev
    
    def test_get_fitness_median(self):
        """Test calculating median fitness across generations."""
        config = MagicMock()
        
        # Generation with 5 members
        genomes = [MockGenome(i, fitness=float(i)) for i in range(1, 6)]
        species1 = MockSpecies(1, genomes)
        population1 = {g.key: g for g in genomes}
        species_set1 = MockSpeciesSet({1: species1})
        self.stats.post_evaluate(config, population1, species_set1, genomes[-1])
        
        medians = self.stats.get_fitness_median()
        self.assertEqual(len(medians), 1)
        self.assertAlmostEqual(medians[0], 3.0)  # Median of [1,2,3,4,5] is 3
    
    def test_best_genome(self):
        """Test retrieving the best genome ever seen."""
        config = MagicMock()
        
        # Add genomes with different fitness values
        for fitness in [5.0, 10.0, 7.0, 3.0, 15.0, 12.0]:
            g = MockGenome(int(fitness), fitness=fitness)
            population = {g.key: g}
            species_set = MockSpeciesSet({1: MockSpecies(1, [g])})
            self.stats.post_evaluate(config, population, species_set, g)
        
        best = self.stats.best_genome()
        self.assertEqual(best.fitness, 15.0)
    
    def test_best_genomes_multiple(self):
        """Test retrieving top N genomes."""
        config = MagicMock()
        
        # Add genomes with different fitness values
        for fitness in [5.0, 10.0, 7.0, 3.0, 15.0, 12.0]:
            g = MockGenome(int(fitness), fitness=fitness)
            population = {g.key: g}
            species_set = MockSpeciesSet({1: MockSpecies(1, [g])})
            self.stats.post_evaluate(config, population, species_set, g)
        
        top_3 = self.stats.best_genomes(3)
        self.assertEqual(len(top_3), 3)
        self.assertEqual(top_3[0].fitness, 15.0)
        self.assertEqual(top_3[1].fitness, 12.0)
        self.assertEqual(top_3[2].fitness, 10.0)
    
    def test_best_genomes_more_than_available(self):
        """Test requesting more best genomes than available."""
        config = MagicMock()
        
        # Add only 2 genomes
        for fitness in [5.0, 10.0]:
            g = MockGenome(int(fitness), fitness=fitness)
            population = {g.key: g}
            species_set = MockSpeciesSet({1: MockSpecies(1, [g])})
            self.stats.post_evaluate(config, population, species_set, g)
        
        top_5 = self.stats.best_genomes(5)
        self.assertEqual(len(top_5), 2)  # Should only return 2
    
    def test_best_unique_genomes(self):
        """Test retrieving unique best genomes (no duplicates by key)."""
        config = MagicMock()
        
        # Add same genome keys multiple times
        g1 = MockGenome(1, fitness=5.0)
        g2 = MockGenome(2, fitness=10.0)
        g1_again = MockGenome(1, fitness=7.0)  # Same key, different fitness
        
        population1 = {g1.key: g1}
        species_set1 = MockSpeciesSet({1: MockSpecies(1, [g1])})
        self.stats.post_evaluate(config, population1, species_set1, g1)
        
        population2 = {g2.key: g2}
        species_set2 = MockSpeciesSet({1: MockSpecies(1, [g2])})
        self.stats.post_evaluate(config, population2, species_set2, g2)
        
        population3 = {g1_again.key: g1_again}
        species_set3 = MockSpeciesSet({1: MockSpecies(1, [g1_again])})
        self.stats.post_evaluate(config, population3, species_set3, g1_again)
        
        # Should have 3 genomes total
        self.assertEqual(len(self.stats.most_fit_genomes), 3)
        
        # But only 2 unique
        unique = self.stats.best_unique_genomes(10)
        self.assertEqual(len(unique), 2)
    
    def test_get_species_sizes(self):
        """Test getting species sizes across generations."""
        config = MagicMock()
        
        # Generation 1: 2 species with 3 and 2 members
        g1, g2, g3 = [MockGenome(i, fitness=float(i)) for i in range(1, 4)]
        g4, g5 = [MockGenome(i, fitness=float(i)) for i in range(4, 6)]
        
        species1 = MockSpecies(1, [g1, g2, g3])
        species2 = MockSpecies(2, [g4, g5])
        population1 = {g.key: g for g in [g1, g2, g3, g4, g5]}
        species_set1 = MockSpeciesSet({1: species1, 2: species2})
        self.stats.post_evaluate(config, population1, species_set1, g3)
        
        # Generation 2: species 1 has 4, species 2 has 0 (extinct)
        g6, g7, g8, g9 = [MockGenome(i, fitness=float(i)) for i in range(6, 10)]
        species3 = MockSpecies(1, [g6, g7, g8, g9])
        population2 = {g.key: g for g in [g6, g7, g8, g9]}
        species_set2 = MockSpeciesSet({1: species3})
        self.stats.post_evaluate(config, population2, species_set2, g9)
        
        sizes = self.stats.get_species_sizes()
        self.assertEqual(len(sizes), 2)  # 2 generations
        self.assertEqual(sizes[0], [3, 2])  # Gen 1: species 1 has 3, species 2 has 2
        self.assertEqual(sizes[1], [4, 0])  # Gen 2: species 1 has 4, species 2 has 0
    
    def test_get_species_fitness(self):
        """Test getting species average fitness across generations."""
        config = MagicMock()
        
        # Generation 1
        g1 = MockGenome(1, fitness=2.0)
        g2 = MockGenome(2, fitness=4.0)
        g3 = MockGenome(3, fitness=10.0)
        
        species1 = MockSpecies(1, [g1, g2])  # Mean fitness = (2.0 + 4.0) / 2 = 3.0
        species2 = MockSpecies(2, [g3])       # Mean fitness = 10.0
        population1 = {g.key: g for g in [g1, g2, g3]}
        species_set1 = MockSpeciesSet({1: species1, 2: species2})
        self.stats.post_evaluate(config, population1, species_set1, g3)
        
        fitness = self.stats.get_species_fitness()
        self.assertEqual(len(fitness), 1)  # 1 generation
        self.assertEqual(len(fitness[0]), 2)  # 2 species
        self.assertAlmostEqual(fitness[0][0], 3.0)  # Mean of fitness [2.0, 4.0]
        self.assertAlmostEqual(fitness[0][1], 10.0)  # Mean of fitness [10.0]
    
    def test_get_species_fitness_with_null_value(self):
        """Test species fitness with custom null value for extinct species."""
        config = MagicMock()
        
        # Generation 1: 2 species
        g1 = MockGenome(1, fitness=5.0)
        g2 = MockGenome(2, fitness=10.0)
        species1 = MockSpecies(1, [g1])
        species2 = MockSpecies(2, [g2])
        population1 = {g.key: g for g in [g1, g2]}
        species_set1 = MockSpeciesSet({1: species1, 2: species2})
        self.stats.post_evaluate(config, population1, species_set1, g2)
        
        # Generation 2: only species 1
        g3 = MockGenome(3, fitness=7.0)
        species3 = MockSpecies(1, [g3])
        population2 = {3: g3}
        species_set2 = MockSpeciesSet({1: species3})
        self.stats.post_evaluate(config, population2, species_set2, g3)
        
        fitness = self.stats.get_species_fitness(null_value='EXTINCT')
        self.assertEqual(fitness[1][1], 'EXTINCT')  # Species 2 extinct in gen 2
    
    def test_save_genome_fitness(self):
        """Test saving genome fitness to CSV file."""
        config = MagicMock()
        
        # Add some generations
        for fitness in [5.0, 10.0, 15.0]:
            g = MockGenome(int(fitness), fitness=fitness)
            population = {g.key: g}
            species_set = MockSpeciesSet({1: MockSpecies(1, [g])})
            self.stats.post_evaluate(config, population, species_set, g)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name
        
        try:
            self.stats.save_genome_fitness(filename=temp_path)
            
            # Read and verify
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), 3)
            # Each line should have best and average fitness
            self.assertIn('5.0', lines[0])
            self.assertIn('10.0', lines[1])
            self.assertIn('15.0', lines[2])
        finally:
            os.unlink(temp_path)
    
    def test_save_species_count(self):
        """Test saving species counts to CSV file."""
        config = MagicMock()
        
        # Generation with 2 species
        g1 = MockGenome(1, fitness=5.0)
        g2 = MockGenome(2, fitness=10.0)
        species1 = MockSpecies(1, [g1])
        species2 = MockSpecies(2, [g2])
        population = {g.key: g for g in [g1, g2]}
        species_set = MockSpeciesSet({1: species1, 2: species2})
        self.stats.post_evaluate(config, population, species_set, g2)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name
        
        try:
            self.stats.save_species_count(filename=temp_path)
            
            # Read and verify
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), 1)  # 1 generation
            # Should have counts for both species
            self.assertIn('1', lines[0])
        finally:
            os.unlink(temp_path)
    
    def test_save_species_fitness(self):
        """Test saving species fitness to CSV file."""
        config = MagicMock()
        
        # Generation with species
        g1 = MockGenome(1, fitness=5.0)
        g2 = MockGenome(2, fitness=7.0)
        species1 = MockSpecies(1, [g1, g2])
        population = {g.key: g for g in [g1, g2]}
        species_set = MockSpeciesSet({1: species1})
        self.stats.post_evaluate(config, population, species_set, g2)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name
        
        try:
            self.stats.save_species_fitness(filename=temp_path)
            
            # Read and verify
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            self.assertEqual(len(lines), 1)
            # Should contain average fitness (5.0 + 7.0) / 2 = 6.0
            self.assertIn('6.0', lines[0])
        finally:
            os.unlink(temp_path)
    
    def test_save_all(self):
        """Test save() method creates all three files."""
        config = MagicMock()
        
        # Add a generation
        g = MockGenome(1, fitness=10.0)
        population = {1: g}
        species_set = MockSpeciesSet({1: MockSpecies(1, [g])})
        self.stats.post_evaluate(config, population, species_set, g)
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Change to temp directory and save
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            self.stats.save()
            os.chdir(original_dir)
            
            # Verify all files exist
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'fitness_history.csv')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'speciation.csv')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'species_fitness.csv')))
        finally:
            # Cleanup
            for fname in ['fitness_history.csv', 'speciation.csv', 'species_fitness.csv']:
                fpath = os.path.join(temp_dir, fname)
                if os.path.exists(fpath):
                    os.unlink(fpath)
            os.rmdir(temp_dir)


if __name__ == '__main__':
    unittest.main()
