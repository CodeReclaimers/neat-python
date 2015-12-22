""" 2-input XOR example """
from __future__ import print_function

from neat import nn, population, visualize

xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
xor_outputs = [0, 1, 1, 0]


def eval_fitness(genomes):
    for g in genomes:
        net = nn.create_feed_forward_phenotype(g)

        error = 0.0
        for inputs, expected in zip(xor_inputs, xor_outputs):
            # Serial activation propagates the inputs through the entire network.
            output = net.serial_activate(inputs)
            error += (output[0] - expected) ** 2

        # When the output matches expected for all inputs, fitness will reach
        # its maximum value of 1.0.
        g.fitness = 1 - error


pop = population.Population('xor2_config')
pop.epoch(eval_fitness, 300)

print('Number of evaluations: {0}'.format(pop.total_evaluations))

# Verify network output against training data.
print('\nBest network output:')
winner = pop.most_fit_genomes[-1]
net = nn.create_feed_forward_phenotype(winner)
for inputs, expected in zip(xor_inputs, xor_outputs):
    output = net.serial_activate(inputs)
    print("expected {0:1.5f} got {1:1.5f}".format(expected, output[0]))

# Visualize the winner network and plot statistics.
visualize.plot_stats(pop.most_fit_genomes, pop.fitness_scores)
visualize.plot_species(pop.species_log)
visualize.draw_net(winner, view=True)
