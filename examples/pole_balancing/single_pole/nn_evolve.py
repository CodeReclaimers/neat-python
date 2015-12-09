"""
Single-pole balancing experiment using a discrete-time recurrent neural network.
"""

from __future__ import print_function
import os
import pickle

from neat import nn, population, visualize
from neat.config import Config
from cart_pole import discrete_actuator_force
from fitness import evaluate_population


# Use the nn network phenotype and the discrete actuator force function.
def fitness_function(genomes):
    evaluate_population(genomes, nn.create_feed_forward_phenotype, discrete_actuator_force)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config = Config(os.path.join(local_dir, 'nn_config'))

pop = population.Population(config)
pop.epoch(fitness_function, 1000)

# Save the winner.
winner = pop.most_fit_genomes[-1]
print('Number of evaluations: %d' % winner.ID)
with open('nn_winner_genome', 'wb') as f:
    pickle.dump(winner, f)

# Plot the evolution of the best/average fitness.
visualize.plot_stats(pop.most_fit_genomes, pop.avg_fitness_scores, ylog=True, filename="nn_fitness.svg")
# Visualizes speciation
visualize.plot_species(pop.species_log, filename="nn_speciation.svg")
# Visualize the best network.
visualize.draw_net(winner, view=True, filename="nn_winner.gv")
