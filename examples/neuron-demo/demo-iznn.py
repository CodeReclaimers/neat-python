from __future__ import print_function

import matplotlib.pyplot as plt
import neat


def plot_spikes(spikes, title):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    fig = plt.figure()
    plt.title("Izhikevich's spiking neuron model u/v ({0!s})".format(title))
    plt.xlabel("Recovery (u)")
    plt.ylabel("Potential (mv)")
    plt.grid()
    plt.plot(u_values, v_values, 'r-')

    plt.show()
    plt.close()


def show(title, a, b, c, d):
    n = neat.iznn.IZNeuron(0.0, a, b, c, d, [])
    spike_train = []
    for i in range(1000):
        n.current = 0.0 if i < 100 or i > 800 else 10.0
        spike_train.append((1.0 * i, n.current, n.v, n.u, n.fired))
        print('{0:d}\t{1:f}\t{2:f}\t{3:f}'.format(i, n.current, n.v, n.u))
        n.advance(0.25)

    plot_spikes(spike_train, title)

show('regular spiking', **neat.iznn.REGULAR_SPIKING_PARAMS)

show('intrinsically bursting', **neat.iznn.INTRINSICALLY_BURSTING_PARAMS)

show('chattering', **neat.iznn.CHATTERING_PARAMS)

show('fast spiking', **neat.iznn.FAST_SPIKING_PARAMS)

show('low-threshold spiking', **neat.iznn.LOW_THRESHOLD_SPIKING_PARAMS)

show('thalamo-cortical', 0.02, 0.25, -65.0, 0.05)

show('resonator', 0.1, 0.26, -65.0, 2.0)

plt.show()