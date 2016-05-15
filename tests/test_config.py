import os

from neat.config import Config


def test_nonexistent_config():
    passed = False
    try:
        c = Config('wubba-lubba-dub-dub')
    except Exception as e:
        passed = 'No such config file' in str(e)
    assert passed


def test_bad_config_activation():
    passed = False
    try:
        local_dir = os.path.dirname(__file__)
        c = Config(os.path.join(local_dir, 'bad_configuration1'))
    except Exception as e:
        passed = 'Invalid activation function name' in str(e)
    assert passed
