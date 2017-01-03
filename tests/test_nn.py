import math
import os
import random

import neat


# def test_concurrent_nn():
#     """This is a stripped-down copy of the `memory` example."""
#
#     # num_tests is the number of random examples each network is tested against.
#     num_tests = 16
#     # N is the length of the test sequence.
#     N = 4
#
#     def eval_fitness(genomes):
#         for g in genomes:
#             net = nn.create_recurrent_phenotype(g)
#
#             error = 0.0
#             for _ in range(num_tests):
#                 # Create a random sequence, and feed it to the network with the
#                 # second input set to zero.
#                 seq = [random.choice((0, 1)) for _ in range(N)]
#                 net.reset()
#                 for s in seq:
#                     inputs = [s, 0]
#                     net.activate(inputs)
#
#                 # Set the second input to one, and get the network output.
#                 for s in seq:
#                     inputs = [0, 1]
#                     output = net.activate(inputs)
#
#                     error += (output[0] - s) ** 2
#
#             g.fitness = -(error / (N * num_tests)) ** 0.5
#
#     # Demonstration of how to add your own custom activation function.
#     def sinc(x):
#         return 1.0 if x == 0 else math.sin(x) / x
#
#     # This sinc function will be available if my_sinc_function is included in the
#     # config file activation_functions option under the pheotype section.
#     # Note that sinc is not necessarily useful for this example, it was chosen
#     # arbitrarily just to demonstrate adding a custom activation function.
#     activation_functions.add('my_sinc_function', sinc)
#
#     local_dir = os.path.dirname(__file__)
#     pop = population.Population(os.path.join(local_dir, 'recurrent_config'))
#     pop.run(eval_fitness, 10)
#
#     # Visualize the winner network and plot/log statistics.
#     # visualize.draw_net(winner, view=True, filename="nn_winner.gv")
#     # visualize.draw_net(winner, view=True, filename="nn_winner-enabled.gv", show_disabled=False)
#     # visualize.draw_net(winner, view=True, filename="nn_winner-enabled-pruned.gv", show_disabled=False, prune_unused=True)
#     # visualize.plot_stats(pop.statistics)
#     # visualize.plot_species(pop.statistics)
#     statistics.save_stats(pop.statistics)
#     statistics.save_species_count(pop.statistics)
#     statistics.save_species_fitness(pop.statistics)
#
#     winner = pop.statistics.best_genome()
#     repr(winner)
#     str(winner)
#     for g in winner.node_genes:
#         repr(g)
#         str(g)
#     for g in winner.conn_genes:
#         repr(g)
#         str(g)
