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
        raise Exception("Did not get a NameError from an unknown activation function")

if __name__ == '__main__':
    test_nonexistent_config()
    #test_bad_config_activation()
    test_bad_config_unknown_option()
