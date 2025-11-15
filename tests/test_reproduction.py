import os
import random
import unittest

import neat
from neat.reporting import ReporterSet
from neat.reproduction import DefaultReproduction


class TestSpawnComputation(unittest.TestCase):
    def test_spawn_adjust1(self):
        adjusted_fitness = [1.0, 0.0]
        previous_sizes = [20, 20]
        pop_size = 40
        min_species_size = 10

        spawn = DefaultReproduction.compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size)
        self.assertEqual(spawn, [27, 13])

        spawn = DefaultReproduction.compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
        self.assertEqual(spawn, [30, 10])

        spawn = DefaultReproduction.compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
        self.assertEqual(spawn, [31, 10])

        spawn = DefaultReproduction.compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
        self.assertEqual(spawn, [31, 10])

    def test_spawn_adjust2(self):
        adjusted_fitness = [0.5, 0.5]
        previous_sizes = [20, 20]
        pop_size = 40
        min_species_size = 10

        spawn = DefaultReproduction.compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size)
        self.assertEqual(spawn, [20, 20])

    def test_spawn_adjust3(self):
        adjusted_fitness = [0.5, 0.5]
        previous_sizes = [30, 10]
        pop_size = 40
        min_species_size = 10

        spawn = DefaultReproduction.compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size)
        self.assertEqual(spawn, [25, 15])

        spawn = DefaultReproduction.compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
        self.assertEqual(spawn, [23, 17])

        spawn = DefaultReproduction.compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
        self.assertEqual(spawn, [21, 19])

        spawn = DefaultReproduction.compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
        self.assertEqual(spawn, [20, 20])

        spawn = DefaultReproduction.compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
        self.assertEqual(spawn, [20, 20])


class TestReproducePopulationSize(unittest.TestCase):
    def test_reproduce_respects_pop_size(self):
        """DefaultReproduction.reproduce must always create exactly pop_size genomes."""
        # Load configuration.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        pop_size = config.pop_size

        # Set up the objects similarly to Population.__init__.
        reporters = ReporterSet()
        stagnation = config.stagnation_type(config.stagnation_config, reporters)
        reproduction = config.reproduction_type(config.reproduction_config,
                                                reporters, stagnation)
        species_set = config.species_set_type(config.species_set_config, reporters)

        # Create an initial population and speciate it.
        population = reproduction.create_new(config.genome_type,
                                             config.genome_config,
                                             pop_size)
        generation = 0
        species_set.speciate(config, population, generation)

        # Run several generations and ensure the population size is invariant.
        random.seed(123)
        for generation in range(5):
            # Assign some non-degenerate fitness values.
            for genome in population.values():
                genome.fitness = random.random()

            population = reproduction.reproduce(config, species_set, pop_size, generation)
            self.assertEqual(len(population), pop_size)

            # Prepare species for the next generation.
            species_set.speciate(config, population, generation + 1)


if __name__ == '__main__':
    unittest.main()
