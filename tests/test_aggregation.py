import os

import neat
from neat import aggregations


# TODO: These tests are just smoke tests to make sure nothing has become badly broken.  Expand
# to include more detailed tests of actual functionality.

##class NotAlmostEqualException(AssertionError):
##    pass


##def assert_almost_equal(a, b):
##    if abs(a - b) > 1e-6:
##        max_abs = max(abs(a), abs(b))
##        abs_rel_err = abs(a - b) / max_abs
##        if abs_rel_err > 1e-6:
##            raise NotAlmostEqualException("{0:.4f} !~= {1:.4f}".format(a, b))


def test_sum():
    assert aggregations.sum_aggregation([1.0, 2.0, 0.5]) == 3.5
    assert aggregations.sum_aggregation([1.0, -1.0, 0.0]) == 0.0


def test_product():
    assert aggregations.product_aggregation([1.0, 2.0, 0.5]) == 1.0
    assert aggregations.product_aggregation([1.0, 0.5, 0.0]) == 0.0


def test_max():
    assert aggregations.max_aggregation([0.0, 1.0, 2.0]) == 2.0
    assert aggregations.max_aggregation([0.0, -1.0, -2.0]) == 0.0


def test_min():
    assert aggregations.min_aggregation([0.0, 1.0, 2.0]) == 0.0
    assert aggregations.min_aggregation([0.0, -1.0, -2.0]) == -2.0


def test_maxabs():
    assert aggregations.maxabs_aggregation([0.0, 1.0, 2.0]) == 2.0
    assert aggregations.maxabs_aggregation([0.0, -1.0, -2.0]) == -2.0


def test_median():
    assert aggregations.median_aggregation([0.0, 1.0, 2.0]) == 1.0
    assert aggregations.median_aggregation([-10.0, 1.0, 3.0, 10.0]) == 2.0


def test_mean():
    assert aggregations.mean_aggregation([0.0, 1.0, 2.0]) == 1.0
    assert aggregations.mean_aggregation([0.0, -1.0, -2.0]) == -1.0


def minabs_aggregation(x):
    """ Not particularly useful - just a check. """
    return min(x, key=abs)


def test_add_minabs():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    config.genome_config.add_aggregation('minabs', minabs_aggregation)
    assert config.genome_config.aggregation_function_defs.get('minabs') is not None
    assert config.genome_config.aggregation_function_defs['minabs'] is not None
    assert config.genome_config.aggregation_function_defs.is_valid('minabs')


def dud_function():
    return 0.0


def test_function_set():
    s = aggregations.AggregationFunctionSet()
    assert s.get('sum') is not None
    assert s.get('product') is not None
    assert s.get('max') is not None
    assert s.get('min') is not None
    assert s.get('maxabs') is not None
    assert s.get('median') is not None
    assert s.get('mean') is not None

    assert s.is_valid('sum')
    assert s.is_valid('product')
    assert s.is_valid('max')
    assert s.is_valid('min')
    assert s.is_valid('maxabs')
    assert s.is_valid('median')
    assert s.is_valid('mean')

    assert not s.is_valid('foo')

    try:
        ignored = s['foo']
    except TypeError:
        pass
    else:
        raise Exception("Should have gotten a TypeError for dict lookup of 'foo'")


def test_bad_add1():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    try:
        config.genome_config.add_aggregation('1.0', 1.0)
    except TypeError:
        pass
    else:
        raise Exception("Should have had a TypeError/derived for 'function' 1.0")


def test_bad_add2():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    try:
        config.genome_config.add_aggregation('dud_function', dud_function)
    except TypeError:
        pass
    else:
        raise Exception("Should have had a TypeError/derived for dud_function")


if __name__ == '__main__':
    test_sum()
    test_product()
    test_max()
    test_min()
    test_maxabs()
    test_median()
    test_mean()
    test_add_minabs()
    test_function_set()
