from __future__ import print_function

import matplotlib.pyplot as plt
import neat


def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u in spikes]
    v_values = [v for t, I, v, u in spikes]
    u_values = [u for t, I, v, u in spikes]
    I_values = [I for t, I, v, u in spikes]

    fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(3, 1, 2)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(3, 1, 3)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def show(title, a, b, c, d):
    n = neat.iznn.IZNeuron(0.0, a, b, c, d, [])
    spike_train = []
    for i in range(1000):
        n.current = 0.0 if i < 100 or i > 800 else 10.0
        spike_train.append((1.0 * i, n.current, n.v, n.u))
        print('{0:d}\t{1:f}\t{2:f}\t{3:f}'.format(i, n.current, n.v, n.u))
        n.advance(0.25)

    plot_spikes(spike_train, view=False, title=title)

show('regular spiking', 0.02, 0.2, -65.0, 8.0)

show('intrinsically bursting', 0.02, 0.2, -55.0, 4.0)

show('chattering', 0.02, 0.2, -50.0, 2.0)

show('fast spiking', 0.1, 0.2, -65.0, 2.0)

show('low-threshold spiking', 0.02, 0.25, -65, 2.0)

show('thalamo-cortical', 0.02, 0.25, -65.0, 0.05)

show('resonator', 0.1, 0.26, -65.0, 2.0)

plt.show()