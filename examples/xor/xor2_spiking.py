""" 2-input XOR example using Izhikevich's spiking neuron model. """
from __future__ import print_function
import os
import matplotlib.pyplot as plt

from neat import population, iznn, visualize
from neat.config import Config


# XOR-2
xor_inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
xor_outputs = (0, 1, 1, 0)

# Maximum amount of simulated time (in milliseconds) we will
# wait for the network to produce an output.
max_time = 50.0
# Izhikevitch parameters for "fast spiking" neurons, simulation time
# step of 1/4 millisecond.
iz_params = [0.1, 0.2, -65.0, 2.0, 0.25]


def compute_output(t0, t1):
    '''Compute the network's output based on the "time to first spike" of the two output neurons.'''
    if t0 is None or t1 is None:
        # If one of the output neurons failed to fire within the allotted time,
        # give a response which produces a large error.
        return -1.0
    else:
        # If the output neurons fire within 1.0 milliseconds of each other,
        # the output is 1, and if they fire more than 11 milliseconds apart,
        # the output is 0, with linear interpolation between 1 and 11 milliseconds.
        response = 1.1 - 0.1 * abs(t0 - t1)
        return max(0.0, min(1.0, response))


def simulate(genome):
    # Create a network of Izhikevitch neurons based on the given genome.
    net = iznn.create_phenotype(genome, *iz_params)
    dt = iz_params[-1]
    error = 0.0
    simulated = []
    for inputData, outputData in zip(xor_inputs, xor_outputs):
        neuron_data = {}
        for i, n in net.neurons.items():
            neuron_data[i] = []

        # Reset the network, apply the XOR inputs, and run for the maximum allowed time.
        net.reset()
        net.set_inputs(inputData)
        t0 = None
        t1 = None
        num_steps = int(max_time / dt)
        for j in range(num_steps):
            t = dt * j
            output = net.advance()

            # Capture the time and neuron membrane potential for later use if desired.
            for i, n in net.neurons.items():
                neuron_data[i].append((t, n.v))

            # Remember time of the first output spikes from each neuron.
            if t0 is None and output[0] > 0:
                t0 = t

            if t1 is None and output[1] > 0:
                t1 = t

        response = compute_output(t0, t1)
        error += (response - outputData) ** 2

        simulated.append((inputData, outputData, t0, t1, neuron_data))

    return error, simulated


def eval_fitness(genomes):
    for genome in genomes:
        error, simulated = simulate(genome)
        genome.fitness = 1 - error


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'xor2_config'))

    # For this network, we use two output neurons and use the difference between
    # the "time to first spike" to determine the network response.  There are
    # probably a great many different choices one could make for an output encoding,
    # and this choice may not be the best for tackling a real problem.
    config.output_nodes = 2

    pop = population.Population(config)
    pop.epoch(eval_fitness, 200)

    winner = pop.most_fit_genomes[-1]
    print('Number of evaluations: {0:d}'.format(winner.ID))

    # Visualize the winner network and plot statistics.
    visualize.plot_stats(pop.most_fit_genomes, pop.avg_fitness_scores)
    visualize.plot_species(pop.species_log)
    visualize.draw_net(winner, view=True)

    # Verify network output against training data.
    print('\nBest network output:')
    net = iznn.create_phenotype(winner, *iz_params)
    error, simulated = simulate(winner)

    # Create a plot of the traces out to the max time for each set of inputs.
    plt.figure(figsize=(12,12))
    for r, (inputData, outputData, t0, t1, neuron_data) in enumerate(simulated):
        response = compute_output(t0, t1)
        print("{0!r} expected {1:.3f} got {2:.3f}".format(inputData, outputData, response))

        plt.subplot(4, 1, r + 1)
        plt.title("Traces for XOR input {{{0:.1f}, {1:.1f}}}".format(*inputData), fontsize=12)
        for i, s in neuron_data.items():
            if i not in net.inputs:
                t, v = zip(*s)
                plt.plot(t, v, "-", label="neuron {0:d}".format(i))

        plt.ylabel("Potential (mv)", fontsize=10)
        plt.ylim(-100, 50)
        plt.tick_params(axis='both', labelsize=8)
        plt.grid()

    plt.xlabel("Time (in ms)", fontsize=10)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig("traces.png", dpi=90)
    plt.show()


if __name__ == '__main__':
    run()
