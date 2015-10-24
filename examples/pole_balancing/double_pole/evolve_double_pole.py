""" Double pole balancing experiment """
import cPickle as pickle
import os

from neat import config, population, chromosome, genome, visualize
from cart_pole import CartPole


def evaluate_population(pop):
    simulation = CartPole(pop, markov=False)
    # comment this line to print the status
    simulation.print_status = False
    simulation.run()


def run():
    # load settings file
    local_dir = os.path.dirname(__file__)
    config.load(os.path.join(local_dir, 'dpole_config'))

    # change the number of inputs accordingly to the type
    # of experiment: markov (6) or non-markov (3)
    # you can also set the configs in dpole_config as long
    # as you have two config files for each type of experiment
    config.Config.input_nodes = 3

    # neuron model type
    chromosome.node_gene_type = genome.NodeGene
    # chromosome.node_gene_type = genome.CTNodeGene

    pop = population.Population()
    pop.epoch(evaluate_population, 200, report=1, save_best=0)

    winner = pop.stats()[0][-1]

    print 'Number of evaluations: %d' % winner.id
    print 'Winner fitness: %f' % winner.fitness

    # save the winner
    with open('winner_chromosome', 'w') as f:
        pickle.dump(winner, f)

    # Plots the evolution of the best/average fitness
    visualize.plot_stats(pop.stats(), ylog=True)
    # Visualizes speciation
    visualize.plot_species(pop.species_log())
    # visualize the best topology
    visualize.draw_net(winner, view=True)


if __name__ == "__main__":
    run()