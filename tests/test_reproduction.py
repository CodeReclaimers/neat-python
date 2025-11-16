import os
import random
import unittest
import copy

import neat
from neat.reporting import ReporterSet
from neat.reproduction import DefaultReproduction
from neat.stagnation import DefaultStagnation
from neat.species import DefaultSpeciesSet


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


class TestElitism(unittest.TestCase):
    """Tests for per-species elitism behavior in DefaultReproduction."""

    def setUp(self):
        # Load standard test configuration.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)

        self.pop_size = self.config.pop_size

        # Set up objects similarly to Population.__init__.
        self.reporters = ReporterSet()
        self.stagnation = DefaultStagnation(self.config.stagnation_config,
                                            self.reporters)
        self.reproduction = DefaultReproduction(self.config.reproduction_config,
                                               self.reporters,
                                               self.stagnation)
        self.species_set = DefaultSpeciesSet(self.config.species_set_config,
                                             self.reporters)

        # Deterministic initial population/speciation.
        random.seed(123)
        self.population = self.reproduction.create_new(
            self.config.genome_type,
            self.config.genome_config,
            self.pop_size,
        )
        self.generation = 0
        self.species_set.speciate(self.config, self.population, self.generation)

    def _assign_deterministic_fitness_by_species(self):
        """Within each species, assign strictly increasing fitness per genome id."""
        for sid, s in self.species_set.species.items():
            members = sorted(s.members.items(), key=lambda kv: kv[0])
            for rank, (gid, genome) in enumerate(members):
                genome.fitness = float(rank + 1)

    @staticmethod
    def _snapshot_genome(genome):
        """Create a primitive snapshot of a genome's structure and parameters.

        This avoids comparing object identity of node/connection genes and focuses
        on their attributes (weights, enabled flags, biases, etc.).
        """
        node_snapshot = {}
        for nid, ng in genome.nodes.items():
            node_snapshot[nid] = {
                'bias': getattr(ng, 'bias', None),
                'response': getattr(ng, 'response', None),
                'activation': getattr(ng, 'activation', None),
                'aggregation': getattr(ng, 'aggregation', None),
            }

        conn_snapshot = {}
        for key, cg in genome.connections.items():
            conn_snapshot[key] = {
                'weight': getattr(cg, 'weight', None),
                'enabled': getattr(cg, 'enabled', None),
                'innovation': getattr(cg, 'innovation', None),
            }

        return {
            'nodes': node_snapshot,
            'connections': conn_snapshot,
            'fitness': genome.fitness,
        }

    def test_elites_preserved_for_surviving_species(self):
        """Elites of non-extinct species are preserved and unmodified between generations.

        For each species that has at least one descendant in the next generation,
        the top min(elitism, species_size) genomes by fitness must survive with
        identical genome parameters.
        """
        self._assign_deterministic_fitness_by_species()

        elitism = self.config.reproduction_config.elitism
        new_ids = None

        # Record original members, elites, and structural snapshots of elite genomes.
        original_members_by_species = {}
        elite_ids_by_species = {}
        elite_snapshot = {}

        for sid, s in self.species_set.species.items():
            members = sorted(s.members.items(), key=lambda kv: kv[1].fitness,
                             reverse=True)
            original_members_by_species[sid] = {gid for gid, _ in members}

            expected_elites = members[:min(elitism, len(members))]
            elite_ids = [gid for gid, _ in expected_elites]
            elite_ids_by_species[sid] = elite_ids

            for gid, genome in expected_elites:
                elite_snapshot[gid] = self._snapshot_genome(genome)

        # Reproduce one generation.
        new_population = self.reproduction.reproduce(
            self.config, self.species_set, self.pop_size, self.generation
        )
        new_ids = set(new_population.keys())

        # For each species, check surviving original genomes.
        for sid, original_ids in original_members_by_species.items():
            surviving_ids = original_ids & new_ids

            if not surviving_ids:
                # Species went extinct; elites may legitimately be gone.
                continue

            expected_elites = elite_ids_by_species[sid]

            # Expect exactly the expected number of elites to survive, and no
            # other original members.
            self.assertEqual(
                surviving_ids,
                set(expected_elites),
                f"Species {sid} survivors {surviving_ids} do not match elites {expected_elites}",
            )

            # Ensure elite genomes were not mutated (structurally or in parameters).
            for gid in expected_elites:
                self.assertIn(gid, new_population)
                self.assertEqual(
                    elite_snapshot[gid],
                    self._snapshot_genome(new_population[gid]),
                    f"Elite genome {gid} from species {sid} was modified",
                )

    def test_elites_may_disappear_only_when_species_extinct(self):
        """Elites are dropped only when their entire species is removed by stagnation."""
        # Need at least two species to distinguish "good" vs "bad".
        if len(self.species_set.species) < 2:
            self.skipTest("Need at least two species for this test")

        # Choose one species to be marked stagnant (bad) and others to improve (good).
        species_items = sorted(self.species_set.species.items(), key=lambda kv: kv[0])
        bad_sid, bad_species = species_items[0]
        other_species_items = species_items[1:]

        # Assign fitness histories and current fitness so that the bad species
        # is considered stagnant while others are improving.
        max_stag = self.config.stagnation_config.max_stagnation
        generation = max_stag + 1

        elitism = self.config.reproduction_config.elitism

        # Record original members and elite ids for all species.
        original_members_by_species = {}
        elite_ids_by_species = {}

        for sid, s in self.species_set.species.items():
            members = sorted(s.members.items(), key=lambda kv: kv[0])
            original_members_by_species[sid] = {gid for gid, _ in members}
            # Use simple per-species ranking for elites.
            elites = members[:min(elitism, len(members))]
            elite_ids_by_species[sid] = [gid for gid, _ in elites]

        # Configure bad species: past improvement and current non-improvement.
        bad_species.fitness_history = [2.0]
        bad_species.last_improved = 0
        for genome in bad_species.members.values():
            genome.fitness = 1.0

        # Configure other species: improvement this generation.
        for sid, s in other_species_items:
            s.fitness_history = [1.0]
            s.last_improved = 0
            for genome in s.members.values():
                genome.fitness = 10.0

        # Reproduce one generation; stagnation logic inside reproduce will
        # remove the bad species entirely.
        new_population = self.reproduction.reproduce(
            self.config, self.species_set, self.pop_size, generation
        )
        new_ids = set(new_population.keys())

        # Bad species should have no surviving original members (extinct).
        bad_original_ids = original_members_by_species[bad_sid]
        bad_survivors = bad_original_ids & new_ids
        self.assertEqual(
            bad_survivors,
            set(),
            f"Bad species {bad_sid} should have gone extinct but has survivors {bad_survivors}",
        )

        # At least one other species must have survivors.
        any_other_survivors = False
        for sid, _ in other_species_items:
            orig_ids = original_members_by_species[sid]
            survivors = orig_ids & new_ids
            if survivors:
                any_other_survivors = True

        self.assertTrue(any_other_survivors, "No surviving species after reproduction")


if __name__ == '__main__':
    unittest.main()
