from __future__ import print_function
from neat import visualize
from neat.iznn import Neuron


def show(title, a, b, c, d):
    n = Neuron(0.0, a, b, c, d)
    spike_train = []
    for i in range(1000):
        n.current = 0.0 if i < 100 or i > 800 else 10.0
        spike_train.append((1.0 * i, n.current, n.v, n.u))
        print('{0:d}\t{1:f}\t{2:f}\t{3:f}'.format(i, n.current, n.v, n.u))
        n.advance()

    visualize.plot_spikes(spike_train, view=True, title=title)

show('regular spiking', 0.02, 0.2, -65.0, 8.0)

show('intrinsically bursting', 0.02, 0.2, -55.0, 4.0)

show('chattering', 0.02, 0.2, -50.0, 2.0)

show('fast spiking', 0.1, 0.2, -65.0, 2.0)

show('low-threshold spiking', 0.02, 0.25, -65, 2.0)

show('thalamo-cortical', 0.02, 0.25, -65.0, 0.05)

show('resonator', 0.1, 0.26, -65.0, 2.0)


