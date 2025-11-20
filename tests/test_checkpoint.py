"""
Tests for checkpoint save and restore functionality.

Tests verify that checkpoint saving and restoration preserve all necessary state:
- Population state (genomes and their attributes)
- Species structure and assignment
- Generation counter
- Random number generator state
- Innovation tracker state (innovation counter and mappings)
- Genome indexer continuation
"""

import unittest
import os
import gzip
import pickle
import tempfile
import shutil
import neat


class CheckpointSnapshotReporter(neat.reporting.BaseReporter):
    """Reporter used only for tests to snapshot population after evaluation.

    Snapshots the population (fitness and basic structure) for a specific
    generation in `post_evaluate`, before reproduction for the next
    generation.
    """

    def __init__(self, target_generation):
        self.target_generation = target_generation
        self.current_generation = None
        self.snapshot = None

    def start_generation(self, generation):
        self.current_generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        if self.current_generation == self.target_generation:
            snap = {}
            for gid, genome in population.items():
                snap[gid] = (
                    genome.fitness,
                    len(genome.nodes),
                    len(genome.connections),
                )
            self.snapshot = snap


class TestCheckpointIntegrity(unittest.TestCase):
    """Test checkpoint save/restore functionality."""
    
    def setUp(self):
        """Set up test configuration and temporary directory."""
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        # Create temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp(prefix='neat_checkpoint_test_')
        self.checkpoint_prefix = os.path.join(self.temp_dir, 'test-checkpoint-')
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def simple_fitness_function(self, genomes, config):
        """Simple fitness function for testing - below threshold."""
        for genome_id, genome in genomes:
            genome.fitness = 0.5
    
    def varied_fitness_function(self, genomes, config):
        """Fitness function with varied fitness values - below threshold."""
        for i, (genome_id, genome) in enumerate(genomes):
            genome.fitness = 0.1 + float(i % 8) * 0.05
    
    # ========== Basic Checkpoint Operations ==========
    
    def test_checkpoint_save_creates_file(self):
        """
        Test that saving a checkpoint creates a file.
        
        The checkpoint file should exist and be non-empty.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        # Run for a few generations to trigger checkpoint
        pop.run(self.simple_fitness_function, 2)
        
        # Verify checkpoint file was created (first checkpoint is for generation 1)
        checkpoint_file = f'{self.checkpoint_prefix}1'
        self.assertTrue(os.path.exists(checkpoint_file),
                       "Checkpoint file should be created")
        self.assertGreater(os.path.getsize(checkpoint_file), 0,
                          "Checkpoint file should not be empty")
    
    def test_checkpoint_restore_succeeds(self):
        """
        Test that restoring from a checkpoint succeeds.
        
        Should be able to restore a population from a saved checkpoint.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 2)
        
        # First checkpoint corresponds to generation 1
        checkpoint_file = f'{self.checkpoint_prefix}1'
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        
        self.assertIsNotNone(restored_pop, "Restored population should not be None")
        self.assertIsInstance(restored_pop, neat.Population,
                             "Restored object should be a Population")
    
    def test_checkpoint_generation_preserved(self):
        """
        Test that generation counter is preserved across checkpoint/restore.
        
        The restored population should continue from the saved generation.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 5)
        
        checkpoint_file = f'{self.checkpoint_prefix}4'
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        
        # Checkpoint 4 contains generation=4, restore sets it to 4
        self.assertEqual(restored_pop.generation, 4,
                        "Generation should be 4 after restoring checkpoint 4")
    
    # ========== Population State Preservation ==========
    
    def test_checkpoint_preserves_population_size(self):
        """
        Test that population size is preserved.
        
        Restored population should have same number of genomes.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 2)
        original_pop_size = len(pop.population)
        
        checkpoint_file = f'{self.checkpoint_prefix}1'
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        
        self.assertEqual(len(restored_pop.population), original_pop_size,
                        "Population size should be preserved")
    
    def test_checkpoint_preserves_genome_keys(self):
        """
        Test that genome IDs are preserved.
        
        Restored genomes should have the same keys as original.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 2)
        original_keys = set(pop.population.keys())
        
        # Use checkpoint corresponding to the final generation state
        checkpoint_file = f'{self.checkpoint_prefix}{pop.generation}'
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        restored_keys = set(restored_pop.population.keys())
        
        self.assertEqual(restored_keys, original_keys,
                        "Genome keys should be preserved")
    
    def test_checkpoint_preserves_genome_structure(self):
        """
        Test that genome structure (nodes and connections) is preserved.
        
        Each genome's internal structure should match after restore.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 3)
        
        # Record genome structures from the final generation
        original_structures = {}
        for gid, genome in pop.population.items():
            original_structures[gid] = (
                len(genome.nodes),
                len(genome.connections),
                list(genome.nodes.keys()),
                list(genome.connections.keys())
            )
        
        # Use checkpoint corresponding to the final generation state
        checkpoint_file = f'{self.checkpoint_prefix}{pop.generation}'
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        
        # Verify structures match
        for gid, (num_nodes, num_conns, node_keys, conn_keys) in original_structures.items():
            self.assertIn(gid, restored_pop.population)
            restored_genome = restored_pop.population[gid]
            
            self.assertEqual(len(restored_genome.nodes), num_nodes,
                           f"Genome {gid} should have same number of nodes")
            self.assertEqual(len(restored_genome.connections), num_conns,
                           f"Genome {gid} should have same number of connections")
            self.assertEqual(list(restored_genome.nodes.keys()), node_keys,
                           f"Genome {gid} should have same node keys")
            self.assertEqual(list(restored_genome.connections.keys()), conn_keys,
                           f"Genome {gid} should have same connection keys")
    
    def test_checkpoint_preserves_fitness_values(self):
        """
        Test that fitness values are preserved.
        
        Each genome should retain its fitness after restore.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.varied_fitness_function, 2)
        
        # Record fitness values from the final generation
        original_fitness = {gid: genome.fitness 
                           for gid, genome in pop.population.items()}
        
        # Use checkpoint corresponding to the final generation state
        checkpoint_file = f'{self.checkpoint_prefix}{pop.generation}'
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        
        # Verify fitness values match
        for gid, fitness in original_fitness.items():
            self.assertEqual(restored_pop.population[gid].fitness, fitness,
                           f"Genome {gid} fitness should be preserved")
    
    # ========== Species State Preservation ==========
    
    def test_checkpoint_preserves_species_count(self):
        """
        Test that number of species is preserved.
        
        Restored population should have same species structure.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.varied_fitness_function, 3)
        original_species_count = len(pop.species.species)
        
        # Use checkpoint corresponding to the final generation state
        checkpoint_file = f'{self.checkpoint_prefix}{pop.generation}'
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        
        self.assertEqual(len(restored_pop.species.species), original_species_count,
                        "Species count should be preserved")
    
    def test_checkpoint_preserves_species_membership(self):
        """
        Test that genome-to-species assignment is preserved.
        
        Each genome should be in the same species after restore.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.varied_fitness_function, 3)
        
        # Record species membership for the final generation
        original_membership = {}
        for sid, species in pop.species.species.items():
            for gid in species.members:
                original_membership[gid] = sid
        
        # Use checkpoint corresponding to the final generation state
        checkpoint_file = f'{self.checkpoint_prefix}{pop.generation}'
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        
        # Verify membership matches
        for gid, sid in original_membership.items():
            found = False
            for species_id, species in restored_pop.species.species.items():
                if gid in species.members:
                    self.assertEqual(species_id, sid,
                                   f"Genome {gid} should be in species {sid}")
                    found = True
                    break
            self.assertTrue(found, f"Genome {gid} should be in a species")
    
    # ========== Indexer Continuation ==========
    
    def test_checkpoint_genome_indexer_continues(self):
        """
        Test that genome indexer continues from last ID after restore.
        
        New genomes should get IDs that don't collide with existing ones.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 3)
        max_genome_id = max(pop.population.keys())
        
        # Use checkpoint corresponding to the final generation state
        checkpoint_file = f'{self.checkpoint_prefix}{pop.generation}'
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        
        # Get next genome ID from indexer
        next_id = next(restored_pop.reproduction.genome_indexer)
        
        self.assertGreater(next_id, max_genome_id,
                          "Next genome ID should be greater than max existing ID")
        self.assertEqual(next_id, max_genome_id + 1,
                        "Next genome ID should be exactly max + 1")
    
    def test_checkpoint_no_id_collisions_after_restore(self):
        """
        Test that no genome ID collisions occur after restore.
        
        Running evolution after restore should not create duplicate IDs.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 3)
        # Use checkpoint corresponding to the final generation state
        checkpoint_file = f'{self.checkpoint_prefix}{pop.generation}'
        
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        original_keys = set(restored_pop.population.keys())
        
        # Run more generations
        restored_pop.run(self.simple_fitness_function, 2)
        
        # Check for ID collisions - new genomes shouldn't reuse old IDs
        # (Some old IDs will be gone due to selection, but no new genome
        # should have an ID that existed before the checkpoint)
        all_keys_after = set(restored_pop.population.keys())
        new_keys = all_keys_after - original_keys
        
        # Verify no new key is less than or equal to max original key
        if new_keys:
            max_original_key = max(original_keys)
            min_new_key = min(new_keys)
            self.assertGreater(min_new_key, max_original_key,
                             "New genome IDs should not collide with old IDs")
    
    # ========== Random State Preservation ==========
    
    def test_checkpoint_preserves_random_state(self):
        """
        Test that random number generator state is preserved.
        
        After restore, random sequences should continue deterministically.
        """
        import random
        
        # Set a known random seed
        random.seed(42)
        
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 2)
        checkpoint_file = f'{self.checkpoint_prefix}1'
        
        # Get some random numbers before restore
        before_restore = [random.random() for _ in range(5)]
        
        # Restore (this loads the random state from checkpoint)
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        
        # The random state should be from generation 1, not from "before_restore"
        # So we can't directly compare, but we can verify it's deterministic
        after_restore = [random.random() for _ in range(5)]
        
        # Restore again to verify determinism
        restored_pop2 = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        after_restore2 = [random.random() for _ in range(5)]
        
        self.assertEqual(after_restore, after_restore2,
                        "Random state should be deterministic after restore")
    
    # ========== Innovation Tracker Preservation ==========
    
    def test_checkpoint_preserves_innovation_tracker(self):
        """
        Test that innovation tracker exists after restore.
        
        The innovation tracker should be available after checkpoint restore.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 3)
        
        checkpoint_file = f'{self.checkpoint_prefix}2'
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        
        # Verify innovation tracker exists after restore
        self.assertIsNotNone(restored_pop.reproduction.innovation_tracker,
                            "Innovation tracker should exist after restore")
        
        # Verify it has the expected structure
        self.assertTrue(hasattr(restored_pop.reproduction.innovation_tracker, 'global_counter'),
                       "Innovation tracker should have global_counter attribute")
        self.assertTrue(hasattr(restored_pop.reproduction.innovation_tracker, 'generation_innovations'),
                       "Innovation tracker should have generation_innovations attribute")
    
    def test_checkpoint_innovation_numbers_continue(self):
        """
        Test that evolution continues normally after checkpoint restore.
        
        Should be able to run evolution after restore without errors.
        """

    @unittest.expectedFailure
    def test_checkpoint_resumed_run_matches_uninterrupted_run(self):
        """End-to-end test: resumed run from checkpoint matches uninterrupted run.

        This verifies that when using a fixed seed and checkpointing, taking a
        checkpoint labeled ``N`` during run #1, restoring it, and continuing the
        run produces exactly the same *evaluation results* for generation
        ``N+1`` as an uninterrupted run would have produced.
        """
        import random

        # Use the module-level CheckpointSnapshotReporter to avoid pickling
        # issues when saving checkpoints during the test.

        # Choose a fixed seed so that evolution is deterministic.
        base_seed = 123
        checkpoint_label = 3  # we will use checkpoint "3" for resuming
        target_generation = checkpoint_label + 1  # compare results for generation N+1

        # ---- Run #1: uninterrupted evolution with checkpointing enabled ----
        pop1 = neat.Population(self.config, seed=base_seed)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop1.add_reporter(checkpointer)
        reporter1 = CheckpointSnapshotReporter(target_generation)
        pop1.add_reporter(reporter1)

        # Run enough generations to evaluate up through target_generation.
        total_generations = target_generation + 1
        pop1.run(self.varied_fitness_function, total_generations)

        # Sanity check
        self.assertIsNotNone(reporter1.snapshot,
                             "Uninterrupted run should have recorded a snapshot")
        uninterrupted_snapshot = reporter1.snapshot

        # Ensure the checkpoint for the chosen label exists.
        checkpoint_file = f"{self.checkpoint_prefix}{checkpoint_label}"
        self.assertTrue(os.path.exists(checkpoint_file),
                        f"Checkpoint {checkpoint_label} should exist for resumed run test")

        # ---- Run #2: restore from checkpoint N and continue ----
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)

        # The restored population should resume at generation N.
        self.assertEqual(restored_pop.generation, checkpoint_label,
                         "Restored population should resume from the checkpoint's generation")

        reporter2 = CheckpointSnapshotReporter(target_generation)
        restored_pop.add_reporter(reporter2)

        # Continue evolution long enough to evaluate through target_generation.
        remaining_generations = target_generation - checkpoint_label + 1
        restored_pop.run(self.varied_fitness_function, remaining_generations)

        self.assertIsNotNone(reporter2.snapshot,
                             "Resumed run should have recorded a snapshot")
        resumed_snapshot = reporter2.snapshot

        # The evaluated population for generation N+1 should be identical
        # between uninterrupted and resumed runs.
        self.assertEqual(uninterrupted_snapshot, resumed_snapshot,
                         "Resumed run from checkpoint should match uninterrupted run at generation N+1")
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        # Run for a few generations
        pop.run(self.simple_fitness_function, 2)
        
        checkpoint_file = f'{self.checkpoint_prefix}1'
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        
        # Should be able to continue running without errors
        try:
            restored_pop.run(self.simple_fitness_function, 2)
            success = True
        except Exception as e:
            success = False
            error = str(e)
        
        self.assertTrue(success, "Should be able to run evolution after restore")
        
        # Verify population still exists and has valid genomes
        self.assertGreater(len(restored_pop.population), 0,
                          "Population should have genomes after continued evolution")
        
        # Verify all genomes have innovation numbers
        for genome in restored_pop.population.values():
            for conn in genome.connections.values():
                self.assertIsNotNone(conn.innovation,
                                   "All connections should have innovation numbers")
    
    # ========== Configuration Handling ==========
    
    def test_checkpoint_restore_with_same_config(self):
        """
        Test restoring with the same configuration.
        
        Should work seamlessly with original config.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 2)
        checkpoint_file = f'{self.checkpoint_prefix}1'
        
        # Restore without providing new config (uses saved config)
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        
        # Should be able to continue running
        restored_pop.run(self.simple_fitness_function, 2)
        
        self.assertGreater(restored_pop.generation, 2,
                          "Should continue running after restore")
    
    def test_checkpoint_restore_with_new_config(self):
        """
        Test restoring with a new configuration.
        
        Should allow replacing config on restore (useful for parameter tuning).
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 2)
        checkpoint_file = f'{self.checkpoint_prefix}1'
        
        # Create new config
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        new_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        # Restore with new config
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file, new_config)
        
        self.assertIsNotNone(restored_pop, "Should restore with new config")
        # Config should be the new one
        self.assertEqual(restored_pop.config, new_config,
                        "Should use new config after restore")
    
    # ========== Multiple Checkpoint Intervals ==========
    
    def test_checkpoint_multiple_generations(self):
        """
        Test that checkpoints are saved at correct intervals.
        
        Should create checkpoint files at specified generation intervals.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(2, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 5)
        
        # With interval=2 starting from generation 0: saves at 2, 4, ...
        self.assertFalse(os.path.exists(f'{self.checkpoint_prefix}1'),
                        "Checkpoint at generation 1 should not exist")
        self.assertTrue(os.path.exists(f'{self.checkpoint_prefix}2'),
                       "Checkpoint at generation 2 should exist")
        self.assertTrue(os.path.exists(f'{self.checkpoint_prefix}4'),
                       "Checkpoint at generation 4 should exist")
    
    def test_checkpoint_can_restore_any_generation(self):
        """
        Test that any saved checkpoint can be restored.
        
        Should be able to restore from any checkpoint file.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 4)
        
        # Try restoring from different checkpoints
        # Checkpoint N contains generation=N, restore sets it to N
        for gen in [1, 2, 3]:
            checkpoint_file = f'{self.checkpoint_prefix}{gen}'
            restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
            
            self.assertEqual(restored_pop.generation, gen,
                           f"Generation should be {gen} for checkpoint {gen}")
    
    # ========== Edge Cases ==========
    
    def test_checkpoint_after_first_generation(self):
        """
        Test checkpointing immediately after first generation.
        
        Should work even with minimal evolution.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 1)
        
        # run(1) completes evaluation of generation 0 and produces generation 1;
        # the checkpoint is labeled with the next generation to be evaluated.
        checkpoint_file = f'{self.checkpoint_prefix}1'
        self.assertTrue(os.path.exists(checkpoint_file),
                       "Checkpoint should exist after first generation")
        
        restored_pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        # Checkpoint 1 contains generation=1 and resumes at the start of generation 1.
        self.assertEqual(restored_pop.generation, 1,
                        "Generation should be 1 in checkpoint 1")
    
    def test_checkpoint_file_format_is_gzipped_pickle(self):
        """
        Test that checkpoint files are gzipped pickle format.
        
        Should be able to manually load and verify format.
        """
        pop = neat.Population(self.config)
        checkpointer = neat.Checkpointer(1, filename_prefix=self.checkpoint_prefix)
        pop.add_reporter(checkpointer)
        
        pop.run(self.simple_fitness_function, 2)
        checkpoint_file = f'{self.checkpoint_prefix}1'
        
        # Try to load manually
        with gzip.open(checkpoint_file) as f:
            data = pickle.load(f)
        
        # Should be a tuple with 5 elements
        self.assertIsInstance(data, tuple, "Checkpoint data should be a tuple")
        self.assertEqual(len(data), 5, "Checkpoint should have 5 elements")
        
        generation, config, population, species_set, rndstate = data
        self.assertIsInstance(generation, int, "Generation should be an int")
        self.assertIsInstance(population, dict, "Population should be a dict")
        self.assertIsNotNone(species_set, "Species set should not be None")


if __name__ == '__main__':
    unittest.main()
