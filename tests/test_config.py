import os

import neat


def test_nonexistent_config():
    """Check that attempting to open a non-existent config file raises
    an Exception with appropriate message."""
    passed = False
    try:
        c = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'wubba-lubba-dub-dub')
    except Exception as e:
        passed = 'No such config file' in str(e)
    assert passed


# Re the below - a partial version is in test_simple_run.py
# TODO: fix this test
# def test_bad_config_activation():
#     """Check that an unknown activation function raises an Exception with
#     the appropriate message."""
#     passed = False
#     try:
#         local_dir = os.path.dirname(__file__)
#         c = Config(os.path.join(local_dir, 'bad_configuration1'))
#     except Exception as e:
#         print(repr(e))
#         passed = 'Invalid activation function name' in str(e)
#     assert passed

def test_bad_config_unknown_option():
    """Check that an unknown option (at least in some sections) raises an exception."""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'bad_configuration2')
    try:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    except NameError:
        pass
    else:
        raise Exception("Did not get a NameError from an unknown configuration file option (in the 'DefaultSpeciesSet' section)")
    config3_path = os.path.join(local_dir, 'bad_configuration3')
    try:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config3_path)
    except NameError:
        pass
    else:
        raise Exception("Did not get a NameError from an unknown configuration file option (in the 'NEAT' section)")


def test_bad_config_RuntimeError(config_file='bad_configuration4'):
    """Test for RuntimeError with a bad configuration file."""
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)
    try:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    except RuntimeError:
        pass
    else:
        raise Exception(
            "Should have had a RuntimeError with {!s}".format(config_file))


def test_bad_config5():
    """Test using bad_configuration5 for a RuntimeError."""
    test_bad_config_RuntimeError(config_file='bad_configuration5')


def test_bad_config6():
    """Test using bad_configuration6 for a RuntimeError."""
    test_bad_config_RuntimeError(config_file='bad_configuration6')


def test_bad_config7():
    """Test using bad_configuration7 for a RuntimeError."""
    test_bad_config_RuntimeError(config_file='bad_configuration7')


def test_bad_config8():
    """Test using bad_configuration8 for a RuntimeError."""
    test_bad_config_RuntimeError(config_file='bad_configuration8')


def test_bad_config9():
    """Test using bad_configuration9 for a RuntimeError."""
    test_bad_config_RuntimeError(config_file='bad_configuration9')


if __name__ == '__main__':
    test_nonexistent_config()
    # test_bad_config_activation()
    test_bad_config_unknown_option()
    test_bad_config_RuntimeError()
    test_bad_config5()
    test_bad_config6()
    test_bad_config7()
    test_bad_config8()
    test_bad_config9()
