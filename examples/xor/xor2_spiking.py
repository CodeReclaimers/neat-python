""" 2-input XOR example using Izhikevich's spiking neuron model. """
from __future__ import print_function

import os

from neat import population, iznn
from neat.config import Config

# Network inputs and expected outputs.
xor_inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
xor_outputs = (0, 1, 1, 0)

# Maximum amount of simulated time (in milliseconds) to wait for the network to produce an output.
max_time = 50.0
# Parameters for "fast spiking" Izhikevich neurons, simulation time step 0.25 millisecond.
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
    # Create a network of Izhikevich neurons based on the given genome.
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

    # Display the winning genome.
    winner = pop.statistics.best_genome()
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    winner = pop.statistics.best_genome()
    print('\nBest genome:\n{!s}'.format(winner))
    print('\nBest network output:')
    net = iznn.create_phenotype(winner, *iz_params)
    dt = iz_params[-1]
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
        print(inputData, response)


if __name__ == '__main__':
    run()
