'''
Single-pole balancing experiment using a continuous-time recurrent neural network (CTRNN).
'''

from __future__ import print_function
import os
import pickle

from neat import ctrnn, genes, population, visualize
from neat.config import Config
from cart_pole import discrete_actuator_force
from fitness import evaluate_population


# Use the CTRNN network phenotype and the discrete actuator force function.
def fitness_function(genomes):
    evaluate_population(genomes, ctrnn.create_phenotype, discrete_actuator_force)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config = Config(os.path.join(local_dir, 'ctrnn_config'))

pop = population.Population(config, node_gene_type=genes.CTNodeGene)
pop.epoch(fitness_function, 2000, report=1, save_best=0)

# Save the winner.
winner = pop.most_fit_genomes[-1]
print('Number of evaluations: %d' % winner.ID)
with open('ctrnn_winner_genome', 'wb') as f:
    pickle.dump(winner, f)

print(winner)

# Plot the evolution of the best/average fitness.
visualize.plot_stats(pop.most_fit_genomes, pop.avg_fitness_scores, ylog=True, filename="ctrnn_fitness.svg")
# Visualizes speciation
visualize.plot_species(pop.species_log, filename="ctrnn_speciation.svg")
# Visualize the best network.
visualize.draw_net(winner, view=True, filename="ctrnn_winner.gv")
