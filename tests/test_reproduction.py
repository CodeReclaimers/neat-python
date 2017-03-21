import sys
import unittest
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
        # NOTE: Due to changes in round() between 2.x and 3.x, this test case
        # has slightly different results based on the Python version.
        if sys.version_info[0] == 3:
            self.assertEqual(spawn, [23, 17])
        else:
            self.assertEqual(spawn, [22, 18])

        spawn = DefaultReproduction.compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
        self.assertEqual(spawn, [21, 19])

        spawn = DefaultReproduction.compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
        self.assertEqual(spawn, [20, 20])

        spawn = DefaultReproduction.compute_spawn(adjusted_fitness, spawn, pop_size, min_species_size)
        self.assertEqual(spawn, [20, 20])


if __name__ == '__main__':
    unittest.main()