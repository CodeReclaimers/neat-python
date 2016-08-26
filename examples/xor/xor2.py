""" 2-input XOR example """
from __future__ import print_function

import os

from neat import nn, population, statistics


# Network inputs and expected outputs.
xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
xor_outputs = [0, 1, 1, 0]

def ideal_demo():
    import neat

    # default Config
    # default to parallel processing using auto-detected # hardware cores
    n = neat.Sequential(xor_inputs, xor_outputs)

    n.evolve(300)

    n.save_statistics(".")

    print('Number of evaluations: {0}'.format(n.total_evaluations))

    # Show output of the most fit genome against training data.
    winner = n.best_genome()
    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')
    winner_output = n.evaluate(winner, xor_inputs)
    for inputs, expected, outputs in zip(xor_inputs, xor_outputs, winner_output):
        print("input {!r}, expected output {0:1.5f} got {1:1.5f}".format(inputs, expected, outputs[0]))



total_evaluations = 0

def eval_fitness(genomes):
    global total_evaluations
    total_evaluations += len(genomes)

    for gid, g in genomes:

        net = nn.create_feed_forward_phenotype(g)

        sum_square_error = 0.0
        for inputs, expected in zip(xor_inputs, xor_outputs):
            # Serial activation propagates the inputs through the entire network.
            output = net.serial_activate(inputs)
            sum_square_error += (output[0] - expected) ** 2

        # When the output matches expected for all inputs, fitness will reach
        # its maximum value of 1.0.
        g.fitness = 1 - sum_square_error


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'xor2_config')
pop = population.Population(config_path)
pop.run(eval_fitness, 300)

# Log statistics.
statistics.save_stats(pop.statistics)
statistics.save_species_count(pop.statistics)
statistics.save_species_fitness(pop.statistics)

print('Number of evaluations: {0}'.format(total_evaluations))

# Show output of the most fit genome against training data.
winner = pop.statistics.best_genome()
print('\nBest genome:\n{!s}'.format(winner))
print('\nOutput:')
winner_net = nn.create_feed_forward_phenotype(winner)
for inputs, expected in zip(xor_inputs, xor_outputs):
    output = winner_net.serial_activate(inputs)
    print("expected {0:1.5f} got {1:1.5f}".format(expected, output[0]))

