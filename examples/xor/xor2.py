import math
import random
import cPickle as pickle
from neat import config, population, chromosome, genome, visualize
from neat.nn import nn_pure as nn
#from neat.nn import nn_cpp as nn # C++ extension

config.load('xor2_config')

# set node gene type
chromosome.node_gene_type = genome.NodeGene

# XOR-2
INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
OUTPUTS = [0, 1, 1, 0]

def eval_fitness(population):
    for chromo in population:
        net = nn.create_ffphenotype(chromo)

        error = 0.0
        #error_stanley = 0.0
        for i, inputs in enumerate(INPUTS):
            net.flush() # not strictly necessary in feedforward nets
            output = net.sactivate(inputs) # serial activation
            error += (output[0] - OUTPUTS[i])**2

            #error_stanley += math.fabs(output[0] - OUTPUTS[i])

        #chromo.fitness = (4.0 - error_stanley)**2 # (Stanley p. 43)
        chromo.fitness = 1 - math.sqrt(error/len(OUTPUTS))

population.Population.evaluate = eval_fitness

pop = population.Population()
pop.epoch(300, report=True, save_best=False)

winner = pop.stats[0][-1]
print 'Number of evaluations: %d' %winner.id

# Visualize the winner network (requires PyDot)
#visualize.draw_net(winner) # best chromosome

# Plots the evolution of the best/average fitness (requires Biggles)
#visualize.plot_stats(pop.stats)
# Visualizes speciation
#visualize.plot_species(pop.species_log)

# Let's check if it's really solved the problem
print '\nBest network output:'
brain = nn.create_ffphenotype(winner)
for i, inputs in enumerate(INPUTS):
    output = brain.sactivate(inputs) # serial activation
    print "%1.5f \t %1.5f" %(OUTPUTS[i], output[0])

# saves the winner
#file = open('winner_chromosome', 'w')
#pickle.dump(winner, file)
#file.close()
