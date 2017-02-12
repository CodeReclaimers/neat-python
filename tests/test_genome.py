import os
import unittest

import neat
from neat.six_util import iterkeys


class TestCreateNew(unittest.TestCase):
    def setUp(self):
        # Determine path to configuration file. This path manipulation is
        # here so that the script will run successfully regardless of the
        # current working directory.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'test_configuration')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)

    def test_unconnected_no_hidden(self):
        gid = 42
        config = self.config.genome_config
        config.initial_connection = 'unconnected'
        config.num_hidden = 0

        g = neat.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(self.config.genome_config)

        print(g)
        self.assertEqual(set(iterkeys(g.nodes)), {0})
        assert(not g.connections)

    def test_unconnected_hidden(self):
        gid = 42
        config = self.config.genome_config
        config.initial_connection = 'unconnected'
        config.num_hidden = 2

        g = neat.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(self.config.genome_config)

        print(g)
        self.assertEqual(set(iterkeys(g.nodes)), {0, 1, 2})
        assert(not g.connections)

    def test_fully_connected_no_hidden(self):
        gid = 42
        config = self.config.genome_config
        config.initial_connection = 'full'
        config.num_hidden = 0

        g = neat.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(config)

        print(g)
        self.assertEqual(set(iterkeys(g.nodes)), {0})
        self.assertEqual(len(g.connections), 2)

        # Check that each input is connected to the output node
        for i in config.input_keys:
            assert((i, 0) in g.connections)


    def test_fully_connected_hidden(self):
        gid = 42
        config = self.config.genome_config
        config.initial_connection = 'full'
        config.num_hidden = 2

        g = neat.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(config)

        print(g)
        self.assertEqual(set(iterkeys(g.nodes)), {0, 1, 2})
        self.assertEqual(len(g.connections), 6)

        # Check that each input is connected to each hidden node.
        for i in config.input_keys:
            for h in (1, 2):
                assert((i, h) in g.connections)

        # Check that each hidden node is connected to the output.
        for h in (1, 2):
            assert((h, 0) in g.connections)

        # Check that inputs are not directly connected to outputs
        for i in config.input_keys:
            assert((i, 0) not in g.connections)

    def test_partially_connected_no_hidden(self):
        gid = 42
        config = self.config.genome_config
        config.initial_connection = 'partial'
        config.connection_fraction = 0.5
        config.num_hidden = 0

        g = neat.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(config)

        print(g)
        self.assertEqual(set(iterkeys(g.nodes)), {0})
        self.assertLess(len(g.connections), 2)

    def test_partially_connected_hidden(self):
        gid = 42
        config = self.config.genome_config
        config.initial_connection = 'partial'
        config.connection_fraction = 0.5
        config.num_hidden = 2

        g = neat.DefaultGenome(gid)
        self.assertEqual(gid, g.key)
        g.configure_new(config)

        print(g)
        self.assertEqual(set(iterkeys(g.nodes)), {0, 1, 2})
        self.assertLess(len(g.connections), 6)


if __name__ == '__main__':
    unittest.main()