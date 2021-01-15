"""
Has the built-in activation functions,
code for using them,
and code for adding new user-defined ones
"""
from __future__ import division
from typing import Callable, Set, Final, Dict, Optional
import math
import types


def sigmoid_activation(z: float) -> float:
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))


def tanh_activation(z: float) -> float:
    z = max(-60.0, min(60.0, 2.5 * z))
    return math.tanh(z)


def sin_activation(z: float) -> float:
    z = max(-60.0, min(60.0, 5.0 * z))
    return math.sin(z)


def gauss_activation(z: float) -> float:
    z = max(-3.4, min(3.4, z))
    return math.exp(-5.0 * z ** 2)


def relu_activation(z: float) -> float:
    return z if z > 0.0 else 0.0


def elu_activation(z: float) -> float:
    return z if z > 0.0 else math.exp(z) - 1


def lelu_activation(z: float) -> float:
    leaky = 0.005
    return z if z > 0.0 else leaky * z


def selu_activation(z: float) -> float:
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return lam * z if z > 0.0 else lam * alpha * (math.exp(z) - 1)


def softplus_activation(z: float) -> float:
    z = max(-60.0, min(60.0, 5.0 * z))
    return 0.2 * math.log(1 + math.exp(z))


def identity_activation(z: float) -> float:
    return z


def clamped_activation(z: float) -> float:
    return max(-1.0, min(1.0, z))


def inv_activation(z: float) -> float:
    try:
        z = 1.0 / z
    except ArithmeticError:  # handle overflows
        return 0.0
    else:
        return z


def log_activation(z: float) -> float:
    z = max(1e-7, z)
    return math.log(z)


def exp_activation(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    return math.exp(z)


def abs_activation(z: float) -> float:
    return abs(z)


def hat_activation(z: float) -> float:
    return max(0.0, 1 - abs(z))


def square_activation(z: float) -> float:
    return z ** 2


def cube_activation(z: float) -> float:
    return z ** 3


class InvalidActivationFunction(TypeError):
    pass


def validate_activation(function: Callable[[float], float]) -> None:
    if not isinstance(function,
                      (types.BuiltinFunctionType,
                       types.FunctionType,
                       types.LambdaType)):
        raise InvalidActivationFunction("A function object is required.")

    if function.__code__.co_argcount != 1:  # avoid deprecated use of `inspect`
        raise InvalidActivationFunction("A single-argument function is required.")


class ActivationFunctionSet(object):
    """
    Contains the list of current valid activation functions,
    including methods for adding and getting them.
    """

    def __init__(self):
        self.functions: Dict[str, Callable[[float], float]] = {}
        self.add('sigmoid', sigmoid_activation)
        self.add('tanh', tanh_activation)
        self.add('sin', sin_activation)
        self.add('gauss', gauss_activation)
        self.add('relu', relu_activation)
        self.add('elu', elu_activation)
        self.add('lelu', lelu_activation)
        self.add('selu', selu_activation)
        self.add('softplus', softplus_activation)
        self.add('identity', identity_activation)
        self.add('clamped', clamped_activation)
        self.add('inv', inv_activation)
        self.add('log', log_activation)
        self.add('exp', exp_activation)
        self.add('abs', abs_activation)
        self.add('hat', hat_activation)
        self.add('square', square_activation)
        self.add('cube', cube_activation)

    def add(self, name: str, function: Callable[[float], float]) -> None:
        validate_activation(function)
        self.functions[name] = function

    def get(self, name: str) -> Optional[Callable[[float], float]]:
        f = self.functions.get(name)
        if f is None:
            raise InvalidActivationFunction("No such activation function: {0!r}".format(name))
        return f

    def is_valid(self, name: str) -> bool:
        return name in self.functions
