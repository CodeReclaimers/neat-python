"""
# drug classification example using a feed-forward network #
"""

import os
import neat
import visualize
import numpy as np
import csv

# 2-input XOR inputs and expected outputs.
with open('data/drug_one_hot.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    data = list(csv_reader)
    node_names = {(i + 1) * -1: name for i, name in enumerate(data[0][:5])}
    node_names.update({i: name for i, name in enumerate(data[0][5:])})
    # remove the header from the data
    data = data[1:]
    num_rows = len(data)
    data = [[float(x) for x in row] for row in data]
    inputs = [row[:5] for row in data]
    outputs = [row[5:11] for row in data]


def select_one(array):
    max_index = np.argmax(array)
    array = np.zeros(len(array))
    array[max_index] = 1

    return array


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi, multi_classification=True)

            genome.fitness -= np.abs(output[0] - xo[0]) + np.abs(output[1] - xo[1]) + np.abs(output[2] - xo[2]) + \
                np.abs(output[3] - xo[3]) + np.abs(output[4] - xo[4])


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 500)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(inputs, outputs):
        output = select_one(winner_net.activate(xi))
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)

