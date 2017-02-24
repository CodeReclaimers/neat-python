import neat
import os

def test_config_save_restore(config_filename_initial = 'test_configuration', config_filename_save = 'save_configuration'):
    """Check if it is possible to restore saved config"""
    passed = False

    # Get config path
    local_dir = os.path.dirname(__file__)
    config_path_initial = os.path.join(local_dir, config_filename_initial)
    config_path_save = os.path.join(local_dir, config_filename_save)

    # Load initial configuration from file
    config_initial = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path_initial)

    # Save configuration to another file
    config_initial.save(config_path_save)

    # Obtain configuration from saved file
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path_save)

    # Test Passed
    passed = True

    assert passed

if __name__ == '__main__':
    test_config_save_restore()
