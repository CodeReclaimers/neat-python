# TODO: fix this and turn it into a real test.
from neat.population import Population
from neat import config, chromosome, genome

config.Config.pop_size = 100
chromosome.node_gene_type = genome.NodeGene


# sample fitness function
def eval_fitness(population):
    for individual in population:
        individual.fitness = 1.0


# creates the population
pop = Population()
# runs the simulation for 250 epochs
pop.epoch(eval_fitness, 250)
