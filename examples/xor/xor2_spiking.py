""" 2-input XOR example using Izhikevich's spiking neuron model. """
from __future__ import print_function
import os
import matplotlib.pyplot as plt
from matplotlib import patches
from neat import population, iznn, visualize
from neat.config import Config

# Network inputs and expected outputs.
xor_inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
xor_outputs = (0, 1, 1, 0)

# Maximum amount of simulated time (in milliseconds) to wait for the network to produce an output.
max_time = 50.0
# Parameters for "fast spiking" Izhikevitch neurons, simulation time step 0.25 millisecond.
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
    sum_square_error = 0.0
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
        v0 = None
        v1 = None
        num_steps = int(max_time / dt)
        for j in range(num_steps):
            t = dt * j
            output = net.advance()

            # Capture the time and neuron membrane potential for later use if desired.
            for i, n in net.neurons.items():
                neuron_data[i].append((t, n.v))

            # Remember time and value of the first output spikes from each neuron.
            if t0 is None and output[0] > 0:
                t0, v0 = neuron_data[net.outputs[0]][-2]

            if t1 is None and output[1] > 0:
                t1, v1 = neuron_data[net.outputs[1]][-2]

        response = compute_output(t0, t1)
        sum_square_error += (response - outputData) ** 2

        simulated.append((inputData, outputData, t0, t1, v0, v1, neuron_data))

    return sum_square_error, simulated


def eval_fitness(genomes):
    for genome in genomes:
        sum_square_error, simulated = simulate(genome)
        genome.fitness = 1 - sum_square_error


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
    pop.run(eval_fitness, 200)

    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    # Visualize the winner network and plot statistics.
    winner = pop.statistics.best_genome()
    node_names = {0: 'A', 1: 'B', 2: 'Out1', 3: 'Out2'}
    visualize.draw_net(winner, view=True, node_names=node_names)
    visualize.plot_stats(pop.statistics)
    visualize.plot_species(pop.statistics)

    # Verify network output against training data.
    print('\nBest network output:')
    net = iznn.create_phenotype(winner, *iz_params)
    sum_square_error, simulated = simulate(winner)

    # Create a plot of the traces out to the max time for each set of inputs.
    plt.figure(figsize=(12, 12))
    for r, (inputData, outputData, t0, t1, v0, v1, neuron_data) in enumerate(simulated):
        response = compute_output(t0, t1)
        print("{0!r} expected {1:.3f} got {2:.3f}".format(inputData, outputData, response))

        axes = plt.subplot(4, 1, r + 1)
        plt.title("Traces for XOR input {{{0:.1f}, {1:.1f}}}".format(*inputData), fontsize=12)
        for i, s in neuron_data.items():
            if i in net.outputs:
                t, v = zip(*s)
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
    run()
