"""
Has the built-in aggregation functions, code for using them,
and code for adding new user-defined ones.
"""

import sys
import types
import warnings

from operator import mul

from neat.math_util import mean, median2

if sys.version_info[0] > 2:
    from functools import reduce


def product_aggregation(x):  # note: `x` is a list or other iterable
    return reduce(mul, x, 1.0)


def sum_aggregation(x):
    return sum(x)


def max_aggregation(x):
    return max(x)


def min_aggregation(x):
    return min(x)


def maxabs_aggregation(x):
    return max(x, key=abs)


def median_aggregation(x):
    return median2(x)


def mean_aggregation(x):
    return mean(x)


class InvalidAggregationFunction(TypeError):
    pass


def validate_aggregation(function):  # TODO: Recognize when need `reduce`
    if not isinstance(function,
                      (types.BuiltinFunctionType,
                       types.FunctionType,
                       types.LambdaType)):
        raise InvalidAggregationFunction("A function object is required.")

    if not (function.__code__.co_argcount >= 1):
        raise InvalidAggregationFunction("A function taking at least one argument is required")


class AggregationFunctionSet(object):
    """Contains aggregation functions and methods to add and retrieve them."""

    def __init__(self):
        self.functions = {}
        self.add('product', product_aggregation)
        self.add('sum', sum_aggregation)
        self.add('max', max_aggregation)
        self.add('min', min_aggregation)
        self.add('maxabs', maxabs_aggregation)
        self.add('median', median_aggregation)
        self.add('mean', mean_aggregation)

    def add(self, name, function):
        validate_aggregation(function)
        self.functions[name] = function

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidAggregationFunction("No such aggregation function: {0!r}".format(name))

        return f

    def __getitem__(self, index):
        warnings.warn("Use get, not indexing ([{!r}]), for aggregation functions".format(index),
                      DeprecationWarning)
        return self.get(index)

    def is_valid(self, name):
        return name in self.functions
