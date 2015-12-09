# -*- coding: UTF-8 -*-
"""
A parallel version of XOR using multiprocessing.Pool.

Since XOR is a simple experiment, a parallel version won't actually
take any advantages of it due to overhead and transfer-communication.
The example below is only a general idea of how to implement a
parallel experiment in neat-python.
"""

from __future__ import print_function
import math
import os
import time
from multiprocessing import Pool

from neat import population, visualize
from neat.config import Config
from neat import nn


# XOR-2
INPUTS = ((0, 0), (0, 1), (1, 0), (1, 1))
OUTPUTS = (0, 1, 1, 0)


def eval_fitness(genomes, pool):
    jobs = []
    for genome in genomes:
        jobs.append(pool.apply_async(parallel_evaluation, (genome,)))

    # assign the fitness back to each genome
    for job, genome in zip(jobs, genomes):
        genome.fitness = job.get(timeout=1)


def parallel_evaluation(genome):
    """ This function will run in parallel """
    net = nn.create_feed_forward_phenotype(genome)

    error = 0.0
    for inputData, outputData in zip(INPUTS, OUTPUTS):
        # serial activation
        output = net.serial_activate(inputData)
        error += (output[0] - outputData) ** 2

    return 1 - math.sqrt(error / len(OUTPUTS))


def run():
    t0 = time.time()

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'xor2_config'))

    num_workers = 6
    pool = Pool(num_workers)
    print("Starting with %d workers" % num_workers)

    def fitness(genomes):
        return eval_fitness(genomes, pool)

    pop = population.Population(config)
    pop.epoch(fitness, 400)

    print("total evolution time %.3f sec" % (time.time() - t0))
    print("time per generation %.3f sec" % ((time.time() - t0) / pop.generation))

    winner = pop.most_fit_genomes[-1]
    print('Number of evaluations: %d' % winner.ID)

    # Verify network output against training data.
    print('\nBest network output:')
    net = nn.create_feed_forward_phenotype(winner)
    for i, inputs in enumerate(INPUTS):
        output = net.serial_activate(inputs)  # serial activation
        print( "%1.5f \t %1.5f" % (OUTPUTS[i], output[0]))

    # Visualize the winner network and plot statistics.
    visualize.plot_stats(pop.most_fit_genomes, pop.avg_fitness_scores)
    visualize.plot_species(pop.species_log)
    visualize.draw_net(winner, view=True)


if __name__ == '__main__':
    run()
