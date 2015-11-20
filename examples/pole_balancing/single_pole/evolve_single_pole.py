""" Single pole balancing experiment """
import os
import cPickle

from neat import population, visualize
from neat.config import Config
from neat.nn import nn_pure as nn
from cart_pole import CartPole, position_limit, angle_limit_radians

num_steps = 10 ** 5
runs_per_net = 5


def evaluate_population(genomes):
    for g in genomes:
        net = nn.create_phenotype(g)

        fitness = 0

        for runs in xrange(runs_per_net):
            sim = CartPole()

            for trials in xrange(num_steps):
                inputs = sim.get_scaled_state()
                action = net.pactivate(inputs)
                # action[0] += 0.4 * (random.random() - 0.5)

                # Apply action to the simulated cart-pole
                force = 10.0 if action[0] > 0.5 else -10.0
                sim.step(force)

                # Stop if the network fails to keep the cart within the position or angle limits.
                # The per-run fitness is the number of time steps the network can balance the pole
                # without exceeding these limits.
                if abs(sim.x) >= position_limit or abs(sim.theta) >= angle_limit_radians:
                    break

                fitness += 1

        # The genome's fitness is its average performance across all runs.
        g.fitness = fitness / float(runs_per_net)


if __name__ == "__main__":
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'spole_config'))

    pop = population.Population(config)
    pop.epoch(evaluate_population, 1000)

    # Save the winner.
    winner = pop.most_fit_genomes[-1]
    print 'Number of evaluations: %d' % winner.ID
    with open('winner_chromosome', 'w') as f:
        cPickle.dump(winner, f)

    # Plot the evolution of the best/average fitness.
    visualize.plot_stats(pop.most_fit_genomes, pop.avg_fitness_scores, ylog=True)
    # Visualizes speciation
    visualize.plot_species(pop.species_log)
    # Visualize the best network.
    visualize.draw_net(winner, view=True)
