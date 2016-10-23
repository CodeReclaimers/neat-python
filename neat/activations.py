import inspect
import math


def sigmoid_activation(z):
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def tanh_activation(z):
    z = max(-60.0, min(60.0, z))
    return math.tanh(z)


def sin_activation(z):
    z = max(-60.0, min(60.0, z))
    return math.sin(z)


def gauss_activation(z):
    z = max(-60.0, min(60.0, z))
    return math.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)


def relu_activation(z):
    return z if z > 0.0 else 0.0


def identity_activation(z):
    return z


def clamped_activation(z):
    return max(-1.0, min(1.0, z))


def inv_activation(z):
    if z == 0:
        return 0.0

    return 1.0 / z


def log_activation(z):
    z = max(1e-7, z)
    return math.log(z)


def exp_activation(z):
    z = max(-60.0, min(60.0, z))
    return math.exp(z)


def abs_activation(z):
    return abs(z)


def hat_activation(z):
    return max(0.0, 1 - abs(z))


def square_activation(z):
    return z ** 2


def cube_activation(z):
    return z ** 3


class InvalidActivationFunction(Exception):
    pass


def validate_activation(function):
    if not inspect.isfunction(function):
        raise InvalidActivationFunction("A function object is required.")

    args = inspect.getargspec(function.__call__)
    if len(args[0]) != 1:
        raise InvalidActivationFunction("A single-argument function is required.")


class ActivationFunctionSet(object):
    def __init__(self):
        self.functions = {'sigmoid': sigmoid_activation,
                          'tanh': tanh_activation,
                          'sin': sin_activation,
                          'gauss': gauss_activation,
                          'relu': relu_activation,
                          'identity': identity_activation,
                          'clamped': clamped_activation,
                          'inv': inv_activation,
                          'log': log_activation,
                          'exp': exp_activation,
                          'abs': abs_activation,
                          'hat': hat_activation,
                          'square': square_activation,
                          'cube': cube_activation}

    def add(self, config_name, function):
        validate_activation(function)
        self.functions[config_name] = function

    def get(self, config_name):
        f = self.functions.get(config_name)
        if f is None:
            raise InvalidActivationFunction("No such function: {0!r}".format(config_name))

        return f

    def is_valid(self, config_name):
        return config_name in self.functions
