import os
import unittest

import neat


class PopulationTests(unittest.TestCase):
    def test_valid_fitness_criterion(self):
        for c in ('max', 'min', 'mean'):
            # Load configuration.
            local_dir = os.path.dirname(__file__)
            config_path = os.path.join(local_dir, 'test_configuration')
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)
            config.fitness_criterion = c

            p = neat.Population(config)

            def eval_genomes(genomes, config):
                for genome_id, genome in genomes:
                    genome.fitness = 1.0

            p.run(eval_genomes, 10)

    def test_invalid_fitness_criterion(self):
        # Load configuration.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        config.fitness_criterion = 'szechaun sauce'

        with self.assertRaises(Exception):
            p = neat.Population(config)

    def test_checkpoint_generation_matches_population_generation(self):
        """When checkpointing every generation, the last checkpoint label should
        match the last evaluated generation.

        Checkpoints are saved after evaluation, so checkpoint ``N`` contains
        the evaluated population for generation ``N``.  After ``run(fn, 5)``
        the loop evaluates generations 0-4 and then reproduces, leaving
        ``p.generation == 5``.  The last checkpoint is for the last evaluated
        generation, which is ``p.generation - 1``.
        """
        # Load configuration.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        p = neat.Population(config)
        filename_prefix = 'neat-checkpoint-test_generation_alignment-'
        # Checkpoint every generation; rely only on generation-based interval
        checkpointer = neat.Checkpointer(1, None, filename_prefix=filename_prefix)
        p.add_reporter(checkpointer)

        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 0.5

        # Run for a fixed number of generations.
        num_generations = 5
        p.run(eval_genomes, num_generations)

        # The last checkpoint is for the last evaluated generation (p.generation - 1),
        # because checkpoints are saved after evaluation but before reproduction.
        self.assertEqual(p.generation - 1, checkpointer.last_generation_checkpoint)

    def test_count_after_checkpoint_restore(self):
        """
        Test that the genome indexer in DefaultGenome continues from the last genome ID
        after restoring from a checkpoint.

        The checkpoint saves the evaluated population (before reproduction),
        so the genome indexer should continue from the max key in that
        checkpoint's population, not the final post-run population.
        """
        # Load configuration.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        p = neat.Population(config)
        filename_prefix = 'neat-checkpoint-test_population'
        checkpointer = neat.Checkpointer(1, 5, filename_prefix=filename_prefix)
        p.add_reporter(checkpointer)

        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 0.5

        p.run(eval_genomes, 5)

        filename = '{0}{1}'.format(
            filename_prefix, checkpointer.last_generation_checkpoint
        )
        restored_population = neat.Checkpointer.restore_checkpoint(filename)
        # The indexer should continue from the max key in the restored population.
        last_genome_key = max(restored_population.population.keys())

        self.assertEqual(
            next(restored_population.reproduction.genome_indexer),
            last_genome_key + 1
        )

    def test_reporter_consistency_after_checkpoint_restore(self):
        """
        Test that ReportSets in the different objects in population are the same
        after restoring from a checkpoint.
        """
        # Load configuration.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

        p = neat.Population(config)
        filename_prefix = 'neat-checkpoint-test_population'
        checkpointer = neat.Checkpointer(1, 5, filename_prefix=filename_prefix)
        p.add_reporter(checkpointer)

        reporter_set = p.reporters
        self.assertEqual(reporter_set, p.reproduction.reporters)
        self.assertEqual(reporter_set, p.species.reporters)

        def eval_genomes(genomes, config):
            for genome_id, genome in genomes:
                genome.fitness = 0.5

        p.run(eval_genomes, 5)

        filename = '{0}{1}'.format(
            filename_prefix, checkpointer.last_generation_checkpoint
        )
        restored_population = neat.Checkpointer.restore_checkpoint(filename)

        # Check that the reporters are consistent
        restored_reporter_set = restored_population.reporters
        self.assertEqual(
            restored_reporter_set,
            restored_population.reproduction.reporters,
            msg="Reproduction reporters do not match after restore"
        )
        self.assertEqual(
            restored_reporter_set,
            restored_population.species.reporters,
            msg="Species reporters do not match after restore"
        )


if __name__ == '__main__':
    pass
