"""
Has the built-in aggregation functions, code for using them,
and code for adding new user-defined ones.
"""

import types
import warnings
from typing import List, Iterable, Callable, Final, Optional, Dict
from functools import reduce
from operator import mul

from neat.math_util import mean, median2


def product_aggregation(x: List[float]) -> float:  # note: `x` is a list or other iterable
    return reduce(mul, x, 1.0)


def sum_aggregation(x: List[float]) -> float:
    return sum(x)


def max_aggregation(x: List[float]) -> float:
    return max(x)


def min_aggregation(x: List[float]) -> float:
    return min(x)


def maxabs_aggregation(x: List[float]) -> float:
    return max(x, key=abs)


def median_aggregation(x: List[float]) -> float:
    return median2(x)


def mean_aggregation(x: List[float]) -> float:
    return mean(x)


class InvalidAggregationFunction(TypeError):
    pass


def validate_aggregation(function: Callable[[List[float]], float]):  # TODO: Recognize when need `reduce`
    if not isinstance(function,
                      (types.BuiltinFunctionType,
                       types.FunctionType,
                       types.LambdaType)):
        raise InvalidAggregationFunction("A function object is required.")

    if not (function.__code__.co_argcount >= 1):
        raise InvalidAggregationFunction("A function taking at least one argument is required")


class AggregationFunctionSet(object):
    """Contains aggregation functions and methods to add and retrieve them."""

    def __init__(self) -> None:
        self.functions: Dict[str, Callable[[List[float]], float]] = {}
        self.add('product', product_aggregation)
        self.add('sum', sum_aggregation)
        self.add('max', max_aggregation)
        self.add('min', min_aggregation)
        self.add('maxabs', maxabs_aggregation)
        self.add('median', median_aggregation)
        self.add('mean', mean_aggregation)

    def add(self, name: str, function: Callable[[List[float]], float]) -> None:
        validate_aggregation(function)
        self.functions[name] = function

    def get(self, name: str) -> Optional[Callable[[List[float]], float]]:
        f: Callable[[List[float]], float] = self.functions.get(name)
        if f is None:
            raise InvalidAggregationFunction("No such aggregation function: {0!r}".format(name))

        return f

    def __getitem__(self, index: str) -> Optional[Callable[[List[float]], float]]:
        warnings.warn("Use get, not indexing ([{!r}]), for aggregation functions".format(index),
                      DeprecationWarning)
        return self.get(index)

    def is_valid(self, name: str) -> bool:
        return name in self.functions
