'''
Single pole balancing experiment
'''

import math
import os
import random
import cPickle

from neat import population, ctrnn, visualize, genes
from neat.config import Config


def cart_pole(net_output, x, x_dot, theta, theta_dot):
    """ Directly copied from Stanley's C++ source code """

    GRAVITY = 9.8
    MASSCART = 1.0
    MASSPOLE = 0.1
    TOTAL_MASS = (MASSPOLE + MASSCART)
    LENGTH = 0.5  # actually half the pole's length
    POLEMASS_LENGTH = (MASSPOLE * LENGTH)
    FORCE_MAG = 10.0
    TAU = 0.02  # seconds between state updates
    FOURTHIRDS = 1.3333333333333

    # force = (net_output - 0.5) * FORCE_MAG * 2
    if net_output > 0.5:
        force = FORCE_MAG
    else:
        force = -FORCE_MAG

    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS

    thetaacc = (GRAVITY * sintheta - costheta * temp) \
               / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta / TOTAL_MASS))

    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    # Update the four state variables, using Euler's method
    x += TAU * x_dot
    x_dot += TAU * xacc
    theta += TAU * theta_dot
    theta_dot += TAU * thetaacc

    return x, x_dot, theta, theta_dot


def evaluate_population(pop):
    twelve_degrees = 0.2094384  # radians
    num_steps = 10 ** 5

    for chromo in pop:

        net = ctrnn.create_phenotype(chromo)

        # old way (harder!)
        # x         = random.uniform(-2.4, 2.4) # cart position, meters
        # x_dot     = random.uniform(-1.0, 1.0) # cart velocity
        # theta     = random.uniform(-0.2, 0.2) # pole angle, radians
        # theta_dot = random.uniform(-1.5, 1.5) # pole angular velocity

        # initial conditions (as used by Stanley)
        x = (random.randint(0, 2 ** 31) % 4800) / 1000.0 - 2.4
        x_dot = (random.randint(0, 2 ** 31) % 2000) / 1000.0 - 1
        theta = (random.randint(0, 2 ** 31) % 400) / 1000.0 - .2
        theta_dot = (random.randint(0, 2 ** 31) % 3000) / 1000.0 - 1.5
        # x = 0.0
        # x_dot = 0.0
        # theta = 0.0
        # theta_dot = 0.0

        fitness = 0

        for trials in xrange(num_steps):
            # maps into [0,1]
            inputs = [(x + 2.4) / 4.8,
                      (x_dot + 0.75) / 1.5,
                      (theta + twelve_degrees) / 0.41,
                      (theta_dot + 1.0) / 2.0]

            action = net.parallel_activate(inputs)

            # Apply action to the simulated cart-pole
            x, x_dot, theta, theta_dot = cart_pole(action[0], x, x_dot, theta, theta_dot)

            # Check for failure.  If so, return steps
            # the number of steps indicates the fitness: higher = better
            fitness += 1
            if abs(x) >= 2.5 or abs(theta) >= twelve_degrees:
                # if abs(theta) > twelve_degrees: # Igel (p. 5)
                # the cart/pole has run/inclined out of the limits
                break

        chromo.fitness = fitness


def run():
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'spole_ctrnn_config'))

    pop = population.Population(config, node_gene_type=genes.CTNodeGene)
    pop.epoch(evaluate_population, 2000, report=1, save_best=0)

    # saves the winner
    winner = pop.most_fit_genomes[-1]
    print 'Number of evaluations: %d' % winner.ID
    with open('winner_chromosome', 'w') as f:
        cPickle.dump(winner, f)

    print winner

    # Plot the evolution of the best/average fitness.
    visualize.plot_stats(pop.most_fit_genomes, pop.avg_fitness_scores, ylog=True, view=True)
    # Visualizes speciation
    visualize.plot_species(pop.species_log)
    # Visualize the best network.
    visualize.draw_net(winner, view=True)


if __name__ == "__main__":
    run()
