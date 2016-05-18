from __future__ import print_function

import os
import pickle
import random

from neat import ctrnn, population
from neat.config import Config
from neat.ctrnn import CTNeuron, Neuron, Network


def test_basic():
    # create two output neurons (they won't receive any external inputs)
    n1 = CTNeuron('OUTPUT', 1, -2.75, 1.0, 'sigmoid', 0.5)
    n1.set_integration_step(0.04)
    repr(n1)
    str(n1)
    n2 = CTNeuron('OUTPUT', 2, -1.75, 1.0, 'sigmoid', 0.5)
    n2.set_integration_step(0.04)
    repr(n2)
    str(n2)
    n1.set_init_state(-0.084000643)
    n2.set_init_state(-0.408035109)

    neurons_list = [n1, n2]
    # create some synapses
    conn_list = [(1, 1, 4.5), (1, 2, -1.0), (2, 1, 1.0), (2, 2, 4.5)]
    # create the network
    net = Network(neurons_list, conn_list, 0)
    net.set_integration_step(0.03)
    repr(net)
    str(net)
    # activates the network
    print("{0:.7f} {1:.7f}".format(n1.output, n2.output))
    outputs = []
    for i in range(1000):
        output = net.parallel_activate()
        outputs.append(output)
        print("{0:.7f} {1:.7f}".format(output[0], output[1]))

    for s in net.synapses:
        repr(s)
        str(s)


def create_simple():
    neurons = [Neuron('INPUT', 1, 0.0, 5.0, 'sigmoid'),
               Neuron('HIDDEN', 2, 0.0, 5.0, 'sigmoid'),
               Neuron('OUTPUT', 3, 0.0, 5.0, 'sigmoid')]
    connections = [(1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.5)]
    map(repr, neurons)

    return Network(neurons, connections, 1)


def test_manual_network():
    net = create_simple()
    repr(net)
    str(net)
    net.serial_activate([0.04])
    net.parallel_activate([0.04])
    repr(net)
    str(net)


def test_evolve():
    test_values = [random.random() for _ in range(10)]

    def evaluate_genome(genomes):
        for g in genomes:
            net = ctrnn.create_phenotype(g)

            fitness = 0.0
            for t in test_values:
                net.reset()
                output = net.serial_activate([t])

                expected = t ** 2

                error = output[0] - expected
                fitness -= error ** 2

            g.fitness = fitness

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'ctrnn_config'))
    config.node_gene_type = ctrnn.CTNodeGene
    config.prob_mutate_time_constant = 0.1
    config.checkpoint_time_interval = 0.1
    config.checkpoint_gen_interval = 1

    pop = population.Population(config)
    pop.run(evaluate_genome, 10)

    # Save the winner.
    print('Number of evaluations: {0:d}'.format(pop.total_evaluations))
    winner = pop.statistics.best_genome()
    with open('winner_genome', 'wb') as f:
        pickle.dump(winner, f)

    repr(winner)
    str(winner)

    for g in winner.node_genes:
        repr(g)
        str(g)
    for g in winner.conn_genes:
        repr(g)
        str(g)
