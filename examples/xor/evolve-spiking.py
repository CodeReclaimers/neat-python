""" 2-input XOR example using Izhikevich's spiking neuron model. """
from __future__ import print_function

import os
from matplotlib import pylab as plt
from matplotlib import patches
import neat

import visualize

# Network inputs and expected outputs.
xor_inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
xor_outputs = (0, 1, 1, 0)

# Maximum amount of simulated time (in milliseconds) to wait for the network to produce an output.
max_time_msec = 20.0


def compute_output(t0, t1):
    '''Compute the network's output based on the "time to first spike" of the two output neurons.'''
    if t0 is None or t1 is None:
        # If neither of the output neurons fired within the allotted time,
        # give a response which produces a large error.
        return -1.0
    else:
        # If the output neurons fire within 1.0 milliseconds of each other,
        # the output is 1, and if they fire more than 11 milliseconds apart,
        # the output is 0, with linear interpolation between 1 and 11 milliseconds.
        response = 1.1 - 0.1 * abs(t0 - t1)
        return max(0.0, min(1.0, response))


def simulate(genome, config):
    # Create a network of "fast spiking" Izhikevich neurons.
    net = neat.iznn.IZNN.create(genome, config)
    dt = net.get_time_step_msec()
    sum_square_error = 0.0
    simulated = []
    for idata, odata in zip(xor_inputs, xor_outputs):
        neuron_data = {}
        for i, n in net.neurons.items():
            neuron_data[i] = []

        # Reset the network, apply the XOR inputs, and run for the maximum allowed time.
        net.reset()
        net.set_inputs(idata)
        t0 = None
        t1 = None
        v0 = None
        v1 = None
        num_steps = int(max_time_msec / dt)
        net.set_inputs(idata)
        for j in range(num_steps):
            t = dt * j
            output = net.advance(dt)

            # Capture the time and neuron membrane potential for later use if desired.
            for i, n in net.neurons.items():
                neuron_data[i].append((t, n.current, n.v, n.u, n.fired))

            # Remember time and value of the first output spikes from each neuron.
            if t0 is None and output[0] > 0:
                t0, I0, v0, u0, f0 = neuron_data[net.outputs[0]][-2]

            if t1 is None and output[1] > 0:
                t1, I1, v1, u1, f0 = neuron_data[net.outputs[1]][-2]

        response = compute_output(t0, t1)
        sum_square_error += (response - odata) ** 2

        #print(genome)
        #visualize.plot_spikes(neuron_data[net.outputs[0]], False)
        #visualize.plot_spikes(neuron_data[net.outputs[1]], True)

        simulated.append((idata, odata, t0, t1, v0, v1, neuron_data))

    return sum_square_error, simulated


def eval_genome(genome, config):
    sum_square_error, simulated = simulate(genome, config)
    return 10.0 - sum_square_error


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(config_path):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config = neat.Config(neat.iznn.IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # For this network, we use two output neurons and use the difference between
    # the "time to first spike" to determine the network response.  There are
    # probably a great many different choices one could make for an output encoding,
    # and this choice may not be the best for tackling a real problem.
    config.output_nodes = 2

    pop = neat.population.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    if 0:
        winner = pop.run(eval_genomes, 3000)
    else:
        pe = neat.ParallelEvaluator(6, eval_genome)
        winner = pop.run(pe.evaluate, 3000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    node_names = {-1:'A', -2: 'B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # Show output of the most fit genome against training data, and create
    # a plot of the traces out to the max time for each set of inputs.
    print('\nBest network output:')
    plt.figure(figsize=(12, 12))
    sum_square_error, simulated = simulate(winner, config)
    for r, (inputData, outputData, t0, t1, v0, v1, neuron_data) in enumerate(simulated):
        response = compute_output(t0, t1)
        print("{0!r} expected {1:.3f} got {2:.3f}".format(inputData, outputData, response))

        axes = plt.subplot(4, 1, r + 1)
        plt.title("Traces for XOR input {{{0:.1f}, {1:.1f}}}".format(*inputData), fontsize=12)
        for i, s in neuron_data.items():
            if i in [0, 1]:
                t, I, v, u, fired = zip(*s)
                plt.plot(t, v, "-", label="neuron {0:d}".format(i))

        # Circle the first peak of each output.
        circle0 = patches.Ellipse((t0, v0), 1.0, 10.0, color='r', fill=False)
        circle1 = patches.Ellipse((t1, v1), 1.0, 10.0, color='r', fill=False)
        axes.add_artist(circle0)
        axes.add_artist(circle1)

        plt.ylabel("Potential (mv)", fontsize=10)
        plt.ylim(-100, 50)
        plt.tick_params(labelsize=8)
        plt.grid()

    plt.xlabel("Time (in ms)", fontsize=10)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig("traces.png", dpi=90)
    plt.show()


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    run(os.path.join(local_dir, 'config-spiking'))
