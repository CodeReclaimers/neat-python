'''Commonly used functions not available in the Python2 standard library.'''
from math import sqrt


def mean(values):
    return sum(map(float, values)) / len(values)


def variance(values):
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def stdev(values):
    return sqrt(variance(values))
