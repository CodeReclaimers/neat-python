'''
Single-pole balancing experiment using a continuous-time recurrent neural network (CTRNN).
'''

from __future__ import print_function

import os
import pickle

import cart_pole

from neat import ctrnn, parallel, population, visualize
from neat.config import Config
from neat.math_util import mean

runs_per_net = 5
num_steps = 60000 # equivalent to 1 minute of simulation time


# Use the CTRNN network phenotype and the discrete actuator force function.
def evaluate_genome(g):
    net = ctrnn.create_phenotype(g)

    fitnesses = []

    for runs in range(runs_per_net):
        sim = cart_pole.CartPole()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        for s in range(num_steps):
            inputs = sim.get_scaled_state()
            action = net.parallel_activate(inputs)

            # Apply action to the simulated cart-pole
            force = cart_pole.discrete_actuator_force(action)
            sim.step(force)

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.
            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break

            fitness += 1.0

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)


# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config = Config(os.path.join(local_dir, 'ctrnn_config'))
config.node_gene_type = ctrnn.CTNodeGene

pop = population.Population(config)
pe = parallel.ParallelEvaluator(4, evaluate_genome)
pop.run(pe.evaluate, 2000)

# Save the winner.
print('Number of evaluations: {0:d}'.format(pop.total_evaluations))
winner = pop.most_fit_genomes[-1]
with open('ctrnn_winner_genome', 'wb') as f:
    pickle.dump(winner, f)

print(winner)

# Plot the evolution of the best/average fitness.
visualize.plot_stats(pop, ylog=True, filename="ctrnn_fitness.svg")
# Visualizes speciation
visualize.plot_species(pop, filename="ctrnn_speciation.svg")
# Visualize the best network.
visualize.draw_net(winner, view=True, filename="ctrnn_winner.gv")
