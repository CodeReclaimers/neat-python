import neat

def test_config_save_restore(config_filename_initial, config_filename_save):
    """Check if it is possible to restore saved config"""
    passed = False
    try:
        # Load initial configuration from file
        config_initial = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation, config_filename_initial)

        # Save configuration to another file
        config_initial.save(config_filename_save)

        # Obtain configuration from saved file
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, config_filename_save)

        # Test Passed
        passed = True
    except:
        pass

    assert passed

if __name__ == '__main__':
    test_config_save_restore('test_configuration', 'save_configuration')
