"""
2-input XOR example -- this is most likely the simplest possible example.
"""

import os
import pandas as pd
import neat
import visualize
from imblearn.over_sampling import SMOTE

import csv

# 2-input XOR inputs and expected outputs.
# with open('data/drug200.csv') as csvfile:
#     csv_reader = csv.reader(csvfile)
#     data = list(csv_reader)
#     # convert the data into a list of floats
#     data = [[float(x) for x in row] for row in data]
#     inputs = [row[:-1] for row in data]
#     outputs = [row[-1] for row in data]

data = pd.read_csv("data/drug200.csv")

#Dividing continuous data into categories
bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
data['Age_binned'] = pd.cut(data['Age'], bins=bin_age, labels=category_age)
data = data.drop(['Age'], axis = 1)

bin_NatoK = [0, 9, 19, 29, 50]
category_NatoK = ['<10', '10-20', '20-30', '>30']
data['Na_to_K_binned'] = pd.cut(data['Na_to_K'], bins=bin_NatoK, labels=category_NatoK)
data = data.drop(['Na_to_K'], axis = 1)
#data = data.iloc[1:, :]
data = pd.get_dummies(data)
data1 = data
data = [[x for x in row] for row in data.values]
inputs = [row[:7]+row[12:] for row in data]
outputs = [row[7:12] for row in data]
# inputs = [data.iloc[:,[0,1,2,4,5]]]
# outputs = [data.loc[:,'Drug']]
#One hot encoding
#data = pd.get_dummies(data)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi)
            # genome.fitness -= (output[0] - int(xo[0])) ** 2
            output_score = 0
            for i in range(len(output)):
                output_score += (output[i] - int(xo[i])) ** 2
            genome.fitness -= output_score/len(output)



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

    node_names = {-1: 'Color', -2: 'Music', -3: 'Alcohol', -4: 'Soda', 0: 'Gender'}
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
