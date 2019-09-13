import os
import unittest

import neat


class ConfigTests(unittest.TestCase):
    def test_config_save_restore(self):
        """Check if it is possible to restore saved config"""

        config_filename_initial = 'test_configuration'
        config_filename_save = 'save_configuration'

        # Get config path
        local_dir = os.path.dirname(__file__)
        config_path_initial = os.path.join(local_dir, config_filename_initial)
        config_path_save = os.path.join(local_dir, config_filename_save)

        # Load initial configuration from file
        config_initial = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path_initial)

        config1 = config_initial.genome_config
        names1 = [p.name for p in config1._params]
        for n in names1:
            assert hasattr(config1, n)

        # Save configuration to another file
        config_initial.save(config_path_save)

        # Obtain configuration from saved file
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path_save)

        config2 = config.genome_config
        names2 = [p.name for p in config2._params]
        for n in names2:
            assert hasattr(config2, n)

        self.assertEqual(names1, names2)

        for n in names1:
            v1 = getattr(config1, n)
            v2 = getattr(config2, n)
            self.assertEqual(v1, v2)

    def test_config_save_restore1(self):
        """Check if it is possible to restore saved config2"""

        config_filename_initial = 'test_configuration2'
        config_filename_save = 'save_configuration2'

        # Get config path
        local_dir = os.path.dirname(__file__)
        config_path_initial = os.path.join(local_dir, config_filename_initial)
        config_path_save = os.path.join(local_dir, config_filename_save)

        # Load initial configuration from file
        config_initial = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path_initial)

        config1 = config_initial.genome_config
        names1 = [p.name for p in config1._params]
        for n in names1:
            assert hasattr(config1, n)

        # Save configuration to another file
        config_initial.save(config_path_save)

        # Obtain configuration from saved file
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path_save)

        config2 = config.genome_config
        names2 = [p.name for p in config2._params]
        for n in names2:
            assert hasattr(config2, n)

        self.assertEqual(names1, names2)

        for n in names1:
            v1 = getattr(config1, n)
            v2 = getattr(config2, n)
            self.assertEqual(v1, v2)

    def test_config_save_error(self):
        """Check if get error on saving bad partial configuration"""

        config_filename_initial = 'test_configuration2'
        config_filename_save = 'save_bad_configuration'

        # Get config path
        local_dir = os.path.dirname(__file__)
        config_path_initial = os.path.join(local_dir, config_filename_initial)
        config_path_save = os.path.join(local_dir, config_filename_save)

        # Load initial configuration from file
        config_initial = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     config_path_initial)

        config_initial.genome_config.connection_fraction = 1.5

        try:
            config_initial.save(config_path_save)
        except RuntimeError:
            pass
        else:
            raise Exception("Did not get RuntimeError on attempt to save bad partial configuration")
