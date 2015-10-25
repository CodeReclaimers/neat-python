""" Single pole balancing experiment """
import random
import os
import math
import cPickle as pickle

from neat import population, visualize
from neat.config import Config
from neat.nn import nn_pure as nn

angle_limit = 30 * math.pi / 180  # radians


def cart_pole(net_output, x, x_dot, theta, theta_dot):
    """ Directly copied from Stanley's C++ source code """

    GRAVITY = 9.8  # m/sec^2
    MASSCART = 1.0  # kg
    MASSPOLE = 0.1  # kg
    TOTAL_MASS = (MASSPOLE + MASSCART)
    LENGTH = 0.5  # meters, actually half the pole's length
    POLEMASS_LENGTH = (MASSPOLE * LENGTH)
    FORCE_MAG = 10.0  # newtons
    TAU = 0.02  # seconds between state updates
    FOURTHIRDS = 1.3333333333333

    # If the net output is 0.5 or greater, apply a force to the
    # right, otherwise apply force to the left.
    force = 0
    if net_output > 0.5:
        force = FORCE_MAG
    elif net_output < 0.5:
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


def evaluate_population(chromosomes):
    num_steps = 10 ** 5

    for chromo in chromosomes:

        net = nn.create_phenotype(chromo)

        # initial conditions (as used by Stanley)        
        x = random.randint(0, 4799) / 1000.0 - 2.4
        x_dot = random.randint(0, 1999) / 1000.0 - 1.0
        theta = random.randint(0, 399) / 1000.0 - 0.2
        theta_dot = random.randint(0, 2999) / 1000.0 - 1.5

        fitness = 0

        for trials in xrange(num_steps):
            # maps into [0,1]
            inputs = [(x + 2.4) / 4.8,
                      (x_dot + 0.75) / 1.5,
                      (theta + angle_limit) / 0.41,
                      (theta_dot + 1.0) / 2.0]

            action = net.pactivate(inputs)
            action[0] += 0.4 * (random.random() - 0.5)

            # Apply action to the simulated cart-pole
            x, x_dot, theta, theta_dot = cart_pole(action[0], x, x_dot, theta, theta_dot)

            # Check for failure.  If so, return steps
            # the number of steps indicates the fitness: higher = better
            fitness += 1
            if abs(x) >= 2.4 or abs(theta) >= angle_limit:
                # if abs(theta) > angle_limit: # Igel (p. 5) uses theta criteria only
                # the cart/pole has run/inclined out of the limits
                break

        chromo.fitness = fitness


if __name__ == "__main__":
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'spole_config'))

    pop = population.Population(config)
    pop.epoch(evaluate_population, 200)

    print 'Number of evaluations: %d' % (pop.stats()[0][-1]).ID

    # Save the winner,
    with open('winner_chromosome', 'w') as f:
        pickle.dump(pop.stats()[0][-1], f)

    # Plot the evolution of the best/average fitness.
    visualize.plot_stats(pop.stats(), ylog=True)
    # Visualizes speciation
    visualize.plot_species(pop.species_log())
    # Visualize the best network.
    visualize.draw_net(pop.stats()[0][-1], view=True)
