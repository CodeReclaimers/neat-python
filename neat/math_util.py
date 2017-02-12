'''Commonly used functions not available in the Python2 standard library.'''
from math import sqrt, exp


def mean(values):
    values = list(values)
    return sum(map(float, values)) / len(values)


def median(values):
    values = list(values)
    values.sort()
    return values[len(values) // 2]


def variance(values):
    values = list(values)
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def stdev(values):
    return sqrt(variance(values))


def softmax(values):
    """
    Compute the softmax of the given value set, v_i = exp(v_i) / s,
    where s = sum(exp(v_0), exp(v_1), ..)."""
    e_values = map(exp, values)
    s = sum(e_values)
    inv_s = 1.0 / s
    return [ev * inv_s for ev in e_values]


# Lookup table for commonly used {value} -> value functions.
stat_functions = {'min': min, 'max': max, 'mean': mean, 'median': median}
