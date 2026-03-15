"""
Has the built-in aggregation functions, code for using them,
and code for adding new user-defined ones.
"""

import inspect
import types
import warnings
from functools import reduce
from operator import mul

from neat.math_util import mean, median2


def product_aggregation(x):  # note: `x` is a list or other iterable
    return reduce(mul, x, 1.0)


def sum_aggregation(x):
    return sum(x)


def max_aggregation(x):
    # Handle empty input (for orphaned nodes with no incoming connections)
    return max(x) if x else 0.0


def min_aggregation(x):
    # Handle empty input (for orphaned nodes with no incoming connections)
    return min(x) if x else 0.0


def maxabs_aggregation(x):
    # Handle empty input (for orphaned nodes with no incoming connections)
    return max(x, key=abs) if x else 0.0


def median_aggregation(x):
    # Handle empty input (for orphaned nodes with no incoming connections)
    return median2(x) if x else 0.0


def mean_aggregation(x):
    # Handle empty input (for orphaned nodes with no incoming connections)
    return mean(x) if x else 0.0


class InvalidAggregationFunction(TypeError):
    pass


def validate_aggregation(function):
    if not callable(function):
        raise InvalidAggregationFunction("A callable object is required.")

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        # CPython builtins (e.g. max, sum) often lack introspectable signatures.
        # Skip signature validation for these; they are assumed correct.
        if isinstance(function, types.BuiltinFunctionType):
            return
        raise InvalidAggregationFunction("Unable to inspect aggregation callable signature.") from exc

    try:
        signature.bind(object())
    except TypeError as exc:
        raise InvalidAggregationFunction(
            "A callable with exactly one required positional argument is required"
        ) from exc


class AggregationFunctionSet:
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
            raise InvalidAggregationFunction(f"No such aggregation function: {name!r}")

        return f

    def __getitem__(self, index):
        warnings.warn(f"Use get, not indexing ([{index!r}]), for aggregation functions",
                      DeprecationWarning)
        return self.get(index)

    def is_valid(self, name):
        return name in self.functions
