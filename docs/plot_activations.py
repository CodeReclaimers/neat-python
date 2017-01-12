import matplotlib.pyplot as plt
import numpy as np
from neat.activations import ActivationFunctionSet

x = np.linspace(-5.0, 5.0, 5000)

afs = ActivationFunctionSet()
for n, f in afs.functions.items():
    plt.figure(figsize=(4, 4))
    plt.plot(x, [f(i) for i in x])
    plt.title(n)
    plt.grid()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect(1)
    plt.savefig('activation-{0}.png'.format(n))
    plt.close()
