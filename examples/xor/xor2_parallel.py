"""
A parallel version of XOR using neat.parallel.

Since XOR is a simple experiment, a parallel version probably won't run any
faster than the single-process version, due to the overhead of
inter-process communication.

If your evaluation function is what's taking up most of your processing time
(and you should probably check by using a profiler while running
single-process), you should see a significant performance improvement by
evaluating in parallel.

This example is only intended to show how to do a parallel experiment
in neat-python.  You can of course roll your own parallelism mechanism
or inherit from ParallelEvaluator if you need to do something more complicated.
"""

from __future__ import print_function

import math
import os
import time

from neat import nn, parallel, population

# Network inputs and expected outputs.
xor_inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
xor_outputs = (0, 1, 1, 0)


def fitness(genome):
    """
    This function will be run in parallel by ParallelEvaluator.  It takes one
    argument (a single genome) and should return one float (that genome's fitness).

    Note that this function needs to be in module scope for multiprocessing.Pool
    (which is what ParallelEvaluator uses) to find it.  Because of this, make
    sure you check for __main__ before executing any code (as we do here in the
    last two lines in the file), otherwise you'll have made a fork bomb
    instead of a neuroevolution demo. :)
    """
    net = nn.create_feed_forward_phenotype(genome)

    sum_square_error = 0.0
    for inputData, outputData in zip(xor_inputs, xor_outputs):
        # serial activation
        output = net.serial_activate(inputData)
        sum_square_error += (output[0] - outputData) ** 2

    return 1 - math.sqrt(sum_square_error / len(xor_outputs))


def run():
    t0 = time.time()

    # Get the path to the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'xor2_config')

    # Use a pool of four workers to evaluate fitness in parallel.
    pe = parallel.ParallelEvaluator(4, fitness)
    pop = population.Population(config_path)
    pop.run(pe.evaluate, 400)

    print("total evolution time {0:.3f} sec".format((time.time() - t0)))
    print("time per generation {0:.3f} sec".format(
        ((time.time() - t0) / pop.generation)))

    print('Number of evaluations: {0:d}'.format(pop.total_evaluations))

    # Show output of the most fit genome against training data.
    winner = pop.statistics.best_genome()
    print('\nBest genome:\n{!s}'.format(winner))
    print('\nBest network output:')
    net = nn.create_feed_forward_phenotype(winner)
    for i, inputs in enumerate(xor_inputs):
        output = net.serial_activate(inputs)  # serial activation
        print("{0:1.5f} \t {1:1.5f}".format(xor_outputs[i], output[0]))


if __name__ == '__main__':
    run()
