"""
This example produces networks that can remember a variable-length sequence of bits. It is
intentionally very (overly?) simplistic just to show the usage of the NEAT library. However,
if you come up with a more interesting or impressive example, please submit a pull request!
"""

from __future__ import print_function

import os
import multiprocessing
import random

import neat
import visualize

# Maximum length of the test sequence.
max_inputs = 4
# Maximum number of ignored inputs
max_ignore = 2
# Number of random examples each network is tested against.
num_tests = 2 ** (max_inputs + max_ignore + 1)


def test_network(net, input_sequence, num_ignore):
    # Feed input bits to the network with the record bit set enabled and play bit disabled.
    net.reset()
    for s in input_sequence:
        inputs = [s, 1.0, 0.0]
        net.activate(inputs)

    # Feed random inputs to be ignored, with both record and play bits disabled.
    for _ in range(num_ignore):
        inputs = [random.choice((0.0, 1.0)), 0.0, 0.0]
        net.activate(inputs)

    # Enable the play bit and get network output.
    outputs = []
    for s in input_sequence:
        inputs = [random.choice((0.0, 1.0)), 0.0, 1.0]
        outputs.append(net.activate(inputs))

    return outputs


def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    error = 0.0
    for _ in range(num_tests):
        num_inputs = random.randint(1, max_inputs)
        num_ignore = random.randint(0, max_ignore)

        # Random sequence of inputs.
        seq = [random.choice((0.0, 1.0)) for _ in range(num_inputs)]

        net.reset()
        outputs = test_network(net, seq, num_ignore)

        for i, o in zip(seq, outputs):
            error += (o[0] - i) ** 2

    return 1.0 - (error / (max_inputs * num_tests))


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate, 1000)

    # Log statistics.
    stats.save()

    # Show output of the most fit genome against a random input.
    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    num_correct = 0
    for n in range(num_tests):
        print('\nRun {0} output:'.format(n))

        num_inputs = random.randint(1, max_inputs)
        num_ignore = random.randint(0, max_ignore)

        seq = [random.choice((0.0, 1.0)) for _ in range(num_inputs)]
        winner_net.reset()
        outputs = test_network(winner_net, seq, num_ignore)

        correct = True
        for i, o in zip(seq, outputs):
            print("\texpected {0:1.5f} got {1:1.5f}".format(i, o[0]))
            correct = correct and round(o[0]) == i
        print("OK" if correct else "FAIL")
        num_correct += 1 if correct else 0

    print("{0} of {1} correct {2:.2f}%".format(num_correct, num_tests, 100.0 * num_correct / num_tests))

    node_names = {-1: 'input', -2: 'record', -3: 'play', 0: 'output'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    run()
