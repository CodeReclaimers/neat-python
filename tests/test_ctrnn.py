from __future__ import print_function

import neat
from neat.activations import sigmoid_activation


def test_basic():
    # Create a fully-connected network of two neurons with no external inputs.
    node1_inputs = [(1, 0.9), (2, 0.2)]
    node2_inputs = [(1, -0.2), (2, 0.9)]

    node_evals = {1: neat.ctrnn.CTRNNNodeEval(0.01, sigmoid_activation, sum, -2.75 / 5.0, 1.0, node1_inputs),
                  2: neat.ctrnn.CTRNNNodeEval(0.01, sigmoid_activation, sum, -1.75 / 5.0, 1.0, node2_inputs)}

    net = neat.ctrnn.CTRNN([], [1, 2], node_evals)

    init1 = 0.0
    init2 = 0.0

    net.set_node_value(1, init1)
    net.set_node_value(2, init2)

    times = [0.0]
    outputs = [[init1, init2]]
    for i in range(1250):
        output = net.advance([], 0.002, 0.002)
        times.append(net.time_seconds)
        outputs.append(output)

#
#
# def create_simple():
#     neurons = [Neuron('INPUT', 1, 0.0, 5.0, 'sigmoid'),
#                Neuron('HIDDEN', 2, 0.0, 5.0, 'sigmoid'),
#                Neuron('OUTPUT', 3, 0.0, 5.0, 'sigmoid')]
#     connections = [(1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.5)]
#     map(repr, neurons)
#
#     return Network(neurons, connections, 1)
#
#
# def test_manual_network():
#     net = create_simple()
#     repr(net)
#     str(net)
#     net.serial_activate([0.04])
#     net.parallel_activate([0.04])
#     repr(net)
#     str(net)
#
#
# def test_evolve():
#     test_values = [random.random() for _ in range(10)]
#
#     def evaluate_genome(genomes):
#         for g in genomes:
#             net = ctrnn.create_phenotype(g)
#
#             fitness = 0.0
#             for t in test_values:
#                 net.reset()
#                 output = net.serial_activate([t])
#
#                 expected = t ** 2
#
#                 error = output[0] - expected
#                 fitness -= error ** 2
#
#             g.fitness = fitness
#
#     # Load the config file, which is assumed to live in
#     # the same directory as this script.
#     local_dir = os.path.dirname(__file__)
#     config = Config(os.path.join(local_dir, 'ctrnn_config'))
#     config.node_gene_type = ctrnn.CTNodeGene
#     config.prob_mutate_time_constant = 0.1
#     config.checkpoint_time_interval = 0.1
#     config.checkpoint_gen_interval = 1
#
#     pop = population.Population(config)
#     pop.run(evaluate_genome, 10)
#
#     # Save the winner.
#     print('Number of evaluations: {0:d}'.format(pop.total_evaluations))
#     winner = pop.statistics.best_genome()
#     with open('winner_genome', 'wb') as f:
#         pickle.dump(winner, f)
#
#     repr(winner)
#     str(winner)
#
#     for g in winner.node_genes:
#         repr(g)
#         str(g)
#     for g in winner.conn_genes:
#         repr(g)
#         str(g)
#
#
if __name__ == '__main__':
    test_basic()
#     test_evolve()
#     test_manual_network()