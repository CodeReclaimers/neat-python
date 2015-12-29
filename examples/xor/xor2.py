""" 2-input XOR example """
from __future__ import print_function

from neat import nn, population, statistics, visualize

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

# Display the most fit genome.
print('\nBest genome:')
winner = pop.most_fit_genomes[-1]
print(winner)

# Verify network output against training data.
print('\nOutput:')
winner_net = nn.create_feed_forward_phenotype(winner)
for inputs, expected in zip(xor_inputs, xor_outputs):
    output = winner_net.serial_activate(inputs)
    print("expected {0:1.5f} got {1:1.5f}".format(expected, output[0]))

# Visualize the winner network and plot/log statistics.
visualize.plot_stats(pop)
visualize.plot_species(pop)
visualize.draw_net(winner, view=True)
statistics.save_stats(pop)
statistics.save_species_count(pop)
statistics.save_species_fitness(pop)