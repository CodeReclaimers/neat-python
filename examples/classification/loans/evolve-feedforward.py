"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import os

import neat
import visualize
import csv

# 2-input XOR inputs and expected outputs.
with open('data/loans.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    data = list(csv_reader)
    # convert the data into a list of floats
    data = [[float(x) for x in row] for row in data]
    inputs = [row[:-1] for row in data]
    outputs = [row[-1] for row in data]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi)
            # genome.fitness -= (output[0] - xo[0]) ** 2
            genome.fitness -= abs(output[0] - xo)  # works better than above


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
    winner = p.run(eval_genomes, 30)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(inputs, outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1: 'Gender', -2: 'Married', -3: 'College Grad', -4: 'Self Employed', -5: 'Income', -6: 'Loan Amt',
                  -7: 'Loan Term', -8: 'Credit History', -9: 'Property Area',  0: 'Approved'}
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
