"""Commonly used statistical helper functions.

These originally filled gaps in the Python 2 standard library and are retained
for API compatibility. Implementations are intentionally simple and avoid
relying on newer stdlib helpers where semantics might differ.
"""

from __future__ import annotations

from math import sqrt, exp
from typing import Callable, Dict, Iterable, List, Sequence


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean of *values* as a float.

    This mirrors the original implementation's behaviour, including
    accepting any value that ``float`` can convert.
    """

    vals: List[float] = [float(v) for v in values]
    return sum(vals) / len(vals)


def median(values: Iterable[float]) -> float:
    """Return the middle value of *values*.

    For even-length inputs this returns the upper-middle element,
    matching the historical behaviour used throughout the project.
    """

    vals = list(values)
    vals.sort()
    return vals[len(vals) // 2]


def median2(values: Iterable[float]) -> float:
    """Median that averages the two middle values for even-length inputs."""

    vals = list(values)
    n = len(vals)
    if n <= 2:
        return mean(vals)

    vals.sort()
    if (n % 2) == 1:
        return vals[n // 2]

    i = n // 2
    return (vals[i - 1] + vals[i]) / 2.0


def variance(values: Iterable[float]) -> float:
    """Population variance of *values*.

    Uses the project-local :func:`mean` helper to preserve historical
    behaviour.
    """

    vals = list(values)
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def stdev(values: Iterable[float]) -> float:
    """Population standard deviation of *values*.

    This is simply ``sqrt(variance(values))``.
    """

    return sqrt(variance(values))


def softmax(values: Iterable[float]) -> List[float]:
    """Compute the softmax of the given value set.

    For each input ``v_i`` this returns ``exp(v_i) / s`` where
    ``s = sum(exp(v_j) for v_j in values)``.
    """

    e_values: List[float] = [exp(v) for v in values]
    s = sum(e_values)
    inv_s = 1.0 / s
    return [ev * inv_s for ev in e_values]


# Lookup table for commonly used {value} -> value functions.
stat_functions: Dict[str, Callable[[Sequence[float]], float]] = {
    'min': min,
    'max': max,
    'mean': mean,
    'median': median,
    'median2': median2,
}
