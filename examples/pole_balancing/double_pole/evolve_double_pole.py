""" Double pole balancing experiment """
import cPickle
import os

from neat import population, visualize
from neat.config import Config
from cart_pole import CartPole


def evaluate_population(pop):
    simulation = CartPole(pop, markov=False)
    # comment this line to print the status
    simulation.print_status = False
    simulation.run()


def run():
    # load settings file
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'dpole_config'))

    # change the number of inputs accordingly to the type
    # of experiment: markov (6) or non-markov (3)
    # you can also set the configs in dpole_config as long
    # as you have two config files for each type of experiment
    config.input_nodes = 3

    pop = population.Population(config)
    pop.epoch(evaluate_population, 200, report=1, save_best=0)

    winner = pop.most_fit_genomes[-1]

    print 'Number of evaluations: {0:d}'.format(winner.ID)
    print 'Winner fitness: {0:f}'.format(winner.fitness)

    # save the winner
    with open('winner_chromosome', 'w') as f:
        cPickle.dump(winner, f)

    # Plots the evolution of the best/average fitness
    visualize.plot_stats(pop, ylog=True)
    # Visualizes speciation
    visualize.plot_species(pop)
    # visualize the best topology
    visualize.draw_net(winner, view=True)


if __name__ == "__main__":
    run()