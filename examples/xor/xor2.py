""" 2-input XOR example """
import math
import os

from neat import population, visualize
from neat.config import Config
from neat.nn import nn_pure as nn


# XOR-2
INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
OUTPUTS = [0, 1, 1, 0]


def eval_fitness(genomes):
    for g in genomes:
        net = nn.create_ffphenotype(g)

        error = 0.0
        for i, inputs in enumerate(INPUTS):
            # Serial activation propagates the inputs through the entire network.
            output = net.sactivate(inputs)
            error += (output[0] - OUTPUTS[i]) ** 2

        g.fitness = 1 - math.sqrt(error / len(OUTPUTS))


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'xor2_config'))

    pop = population.Population(config)
    pop.epoch(eval_fitness, 300)

    winner = pop.most_fit_genomes[-1]
    print 'Number of evaluations: %d' % winner.ID

    # Verify network output against training data.
    print '\nBest network output:'
    net = nn.create_ffphenotype(winner)
    for i, inputs in enumerate(INPUTS):
        output = net.sactivate(inputs)  # serial activation
        print "%1.5f \t %1.5f" % (OUTPUTS[i], output[0])

    # Visualize the winner network and plot statistics.
    visualize.plot_stats(pop.most_fit_genomes, pop.avg_fitness_scores)
    visualize.plot_species(pop.species_log)
    visualize.draw_net(winner, view=True)


if __name__ == '__main__':
    run()