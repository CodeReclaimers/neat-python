import math
from neat import config, population, chromosome, genome, iznn, visualize
#from psyco.classes import *

config.load('xor2_config')

# Temporary workaround
chromosome.node_gene_type = genome.NodeGene

# For spiking networks
config.Config.output_nodes = 2

# XOR-2
INPUTS = ((0, 0), (0, 1), (1, 0), (1, 1))
OUTPUTS = (0, 1, 1, 0)

# For how long are we going to wait for an answer from the network?
MAX_TIME = 100 # in miliseconds of simulation time

def eval_fitness(population):
    for chromosome in population:
        brain = iznn.create_phenotype(chromosome)
        error = 0.0
        for i, input in enumerate(INPUTS):
            for j in range(MAX_TIME):
                output = brain.advance([x * 10 for x in input])
                if output != [False, False]:
                    break
            if output[0] and not output[1]: # Network answered 1
                error += (1 - OUTPUTS[i])**2
            elif not output[0] and output[1]: # Network answered 0
                error += (0 - OUTPUTS[i])**2
            else: # No answer or ambiguous
                error += 1
        chromosome.fitness = 1 - math.sqrt(error/len(OUTPUTS))
        if not chromosome.fitness:
            chromosome.fitness = 0.00001

population.Population.evaluate = eval_fitness
pop = population.Population()
pop.epoch(200, report=True, save_best=0)

# Draft solution for network visualizing
visualize.draw_net(pop.stats[0][-1]) # best chromosome
# Plots the evolution of the best/average fitness
visualize.plot_stats(pop.stats)
# Visualizes speciation
#visualize.plot_species(pop.species_log)

# saves the winner
#file = open('winner_chromosome', 'w')
#pickle.dump(pop.stats[0][-1], file)
#file.close()
