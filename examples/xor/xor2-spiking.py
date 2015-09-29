import math
import os

from neat import config, population, chromosome, genome, iznn, visualize

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config.load(os.path.join(local_dir, 'xor2_config'))

# Temporary workaround
chromosome.node_gene_type = genome.NodeGene

# For spiking networks
config.Config.output_nodes = 2

# XOR-2
INPUTS = ((0, 0), (0, 1), (1, 0), (1, 1))
OUTPUTS = (0, 1, 1, 0)

# Length of simulated time (in milliseconds) for which will
# wait for the network to produce an output.
MAX_TIME = 100

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
            else:
                # No answer or ambiguous
                error += 1
        chromosome.fitness = 1 - math.sqrt(error/len(OUTPUTS))
        if not chromosome.fitness:
            chromosome.fitness = 0.00001

pop = population.Population()
pop.epoch(eval_fitness, 200, report=True, save_best=False)

# Draft solution for network visualizing
visualize.draw_net(pop.stats()[0][-1])
# Plots the evolution of the best/average fitness
visualize.plot_stats(pop.stats())
# Visualizes speciation
visualize.plot_species(pop.species_log())

