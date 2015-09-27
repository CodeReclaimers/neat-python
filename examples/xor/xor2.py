""" 2-input XOR example """
import math
import os

from neat import config, population, chromosome, genome, visualize
from neat.nn import nn_pure as nn

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config.load(os.path.join(local_dir, 'xor2_config'))

# set node gene type
chromosome.node_gene_type = genome.NodeGene

# XOR-2
INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
OUTPUTS = [0, 1, 1, 0]

def eval_fitness(population):
    for chromo in population:
        net = nn.create_ffphenotype(chromo)

        error = 0.0
        for i, inputs in enumerate(INPUTS):
            # flush is not strictly necessary in feedforward nets.
            net.flush()
            # serial activation
            output = net.sactivate(inputs)
            error += (output[0] - OUTPUTS[i])**2

        chromo.fitness = 1 - math.sqrt(error/len(OUTPUTS))

population.Population.evaluate = eval_fitness

pop = population.Population()
pop.epoch(300, report=True, save_best=False)

winner = pop.stats[0][-1]
print 'Number of evaluations: %d' %winner.id

# Visualize the winner network (requires PyDot)
#visualize.draw_net(winner) # best chromosome

# Plots the evolution of the best/average fitness (requires Biggles)
visualize.plot_stats(pop.stats)
# Visualizes speciation
visualize.plot_species(pop.species_log)

# Let's check if it's really solved the problem
print '\nBest network output:'
brain = nn.create_ffphenotype(winner)
for i, inputs in enumerate(INPUTS):
    output = brain.sactivate(inputs) # serial activation
    print "%1.5f \t %1.5f" %(OUTPUTS[i], output[0])
