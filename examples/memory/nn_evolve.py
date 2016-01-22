"""
This example produces networks that can remember a fixed-length sequence of bits. It is
intentionally very (overly?) simplistic just to show the usage of the NEAT library. However,
if you come up with a more interesting or impressive example, please submit a pull request!
"""

from __future__ import print_function

import math
import os
import random

from neat import nn, population, statistics, visualize, parallel, activation_functions

# num_tests is the number of random examples each network is tested against.
num_tests = 16
# N is the length of the test sequence.
N = 4


def eval_fitness(g):
    net = nn.create_recurrent_phenotype(g)

    error = 0.0
    for _ in range(num_tests):
        # Create a random sequence, and feed it to the network with the
        # second input set to zero.
        seq = [random.choice((0, 1)) for _ in range(N)]
        net.reset()
        for s in seq:
            inputs = [s, 0]
            net.activate(inputs)

        # Set the second input to one, and get the network output.
        for s in seq:
            inputs = [0, 1]
            output = net.activate(inputs)

            error += (output[0] - s) ** 2

    return -(error / (N * num_tests)) ** 0.5


# Demonstration of how to add your own custom activation function.
def sinc(x):
    return 1.0 if x == 0 else math.sin(x) / x

# This sinc function will be available if my_sinc_function is included in the
# config file activation_functions option under the pheotype section.
# Note that sinc is not necessarily useful for this example, it was chosen
# arbitrarily just to demonstrate adding a custom activation function.
activation_functions.add('my_sinc_function', sinc)


local_dir = os.path.dirname(__file__)
pop = population.Population(os.path.join(local_dir, 'nn_config'))
pe = parallel.ParallelEvaluator(4, eval_fitness)
pop.run(pe.evaluate, 1000)

print('Number of evaluations: {0}'.format(pop.total_evaluations))

# Display the most fit genome.
print('\nBest genome:')
winner = pop.most_fit_genomes[-1]
print(winner)

# Verify network output against a few randomly-generated sequences.
winner_net = nn.create_recurrent_phenotype(winner)
for n in range(4):
    print('\nRun {0} output:'.format(n))
    seq = [random.choice((0, 1)) for _ in range(N)]
    winner_net.reset()
    for s in seq:
        winner_net.activate([s, 0])

    for s in seq:
        output = winner_net.activate([0, 1])
        print("expected {0:1.5f} got {1:1.5f}".format(s, output[0]))

# Visualize the winner network and plot/log statistics.
visualize.draw_net(winner, view=True, filename="nn_winner.gv")
visualize.draw_net(winner, view=True, filename="nn_winner-enabled.gv", show_disabled=False)
visualize.draw_net(winner, view=True, filename="nn_winner-enabled-pruned.gv", show_disabled=False, prune_unused=True)
visualize.plot_stats(pop)
visualize.plot_species(pop)
statistics.save_stats(pop)
statistics.save_species_count(pop)
statistics.save_species_fitness(pop)