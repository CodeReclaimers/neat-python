import math


def sigmoid_activation(bias, response, x):
    z = bias + x * response
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def tanh_activation(bias, response, x):
    z = bias + x * response
    z = max(-60.0, min(60.0, z))
    return math.tanh(z)


def sin_activation(bias, response, x):
    z = bias + x * response
    z = max(-60.0, min(60.0, z))
    return math.sin(z)


def gauss_activation(bias, response, x):
    z = bias + x * response
    z = max(-60.0, min(60.0, z))
    return math.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)


def relu_activation(bias, response, x):
    z = bias + x * response
    return z if z > 0.0 else 0


def identity_activation(bias, response, x):
    return bias + x * response


def clamped_activation(bias, response, x):
    z = bias + x * response
    return max(-1.0, min(1.0, z))


def inv_activation(bias, response, x):
    z = bias + x * response
    if z == 0:
        return 0.0

    return 1.0 / z


def log_activation(bias, response, x):
    z = bias + x * response
    z = max(1e-7, z)
    return math.log(z)


def exp_activation(bias, response, x):
    z = bias + x * response
    z = max(-60.0, min(60.0, z))
    return math.exp(z)


def abs_activation(bias, response, x):
    z = bias + x * response
    return abs(z)


def hat_activation(bias, response, x):
    z = bias + x * response
    return max(0.0, 1 - abs(z))


def square_activation(bias, response, x):
    z = bias + x * response
    return z ** 2


def cube_activation(bias, response, x):
    z = bias + x * response
    return z ** 3


activations = {'sigmoid':sigmoid_activation,
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