from __future__ import print_function

import os
from neat import nn, population, statistics
from neat.config import Config


def test_run():
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

    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'test_configuration'))

    pop = population.Population(config)
    pop.run(eval_fitness, 10)

    winner = pop.statistics.best_genome()

    # Validate winner.
    for g in pop.statistics.most_fit_genomes:
        assert winner.fitness >= g.fitness

    statistics.save_stats(pop.statistics)
    statistics.save_species_count(pop.statistics)
    statistics.save_species_fitness(pop.statistics)

