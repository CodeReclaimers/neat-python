# ******************************** #
# Double pole balancing experiment #
# ******************************** #
from neat import config, population, chromosome, genome2, visualize
from neat import nn
import math, random
import cPickle as pickle
from cart_pole import CartPole

def evaluate_population(population):
    
    num_steps = 1500
    
    for chromo in population:
        
        net = nn.create_phenotype(chromo)        
        # initialize the cart-pole experiment
        cart = CartPole(net, markov=True)
        # evaluates network performance
        fitness = cart.evaluate(num_steps)
        # set chromosome fitness
        chromo.fitness = fitness

if __name__ == "__main__":
    
    config.load('dpole_config') 

    # Temporary workaround
    chromosome.node_gene_type = genome2.NodeGene
    
    population.Population.evaluate = evaluate_population
    pop = population.Population()
    pop.epoch(200, stats=1, save_best=0)
    
    # visualize the best topology
    #visualize.draw_net(pop.stats[0][-1]) # best chromosome
    # Plots the evolution of the best/average fitness
    #visualize.plot_stats(pop.stats)
    
    # saves the winner
    file = open('winner_chromosome', 'w')
    pickle.dump(pop.stats[0][-1], file)
    file.close()
