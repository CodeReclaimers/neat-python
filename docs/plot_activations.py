import matplotlib.pyplot as plt
import numpy as np
from neat.activations import *

x=np.linspace(-5.0, 5.0, 5000)

afs = ActivationFunctionSet()
for n, f in afs.functions.items():
    plt.figure(figsize=(4,4))
    plt.plot(x, [f(i) for i in x])
    plt.title(n)
    plt.grid()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect(1)
    plt.savefig('activation-{0}.png'.format(n))
    plt.close()

if 0:
    plt.plot(x, [sigmoid_activation(i) for i in x], label='sigmoid')
    # plt.plot(x, [tanh_activation(i) for i in x], label='tanh')
    # plt.plot(x, [sin_activation(i) for i in x], label='sin')
    # plt.plot(x, [gauss_activation(i) for i in x], label='gauss')
    plt.plot(x, [relu_activation(i) for i in x], label='relu')
    # plt.plot(x, [identity_activation(i) for i in x], label='identity')
    plt.plot(x, [clamped_activation(i) for i in x], label='clamped')
    # plt.plot(x, [inv_activation(i) for i in x], label='inv')
    # plt.plot(x, [log_activation(i) for i in x], label='log')
    # plt.plot(x, [exp_activation(i) for i in x], label='exp')
    # plt.plot(x, [abs_activation(i) for i in x], label='abs')
    # plt.plot(x, [hat_activation(i) for i in x], label='hat')
    # plt.plot(x, [square_activation(i) for i in x], label='square')
    # plt.plot(x, [cube_activation(i) for i in x], label='cube')
    plt.plot(x, [softplus_activation(i) for i in x], label='softplus')
    plt.grid()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.legend(loc='best')
    plt.gca().set_aspect(1)
    plt.show()