# -*- coding: UTF-8 -*-
from __future__ import print_function
import warnings
import csv

try:
    import numpy as np
except ImportError:
    np = None
    warnings.warn('Could not import optional dependency NumPy.')

def save_stats(best_genomes, avg_scores, ylog=False, view=False, filename='fitness_history.csv'):
    """ Saves the population's average and best fitness. """
    csvfile = open(filename, 'wb')
    statWriter = csv.writer(csvfile, delimiter=' ')

    generation = range(len(best_genomes))
    fitness = [c.fitness for c in best_genomes]

    for i in generation:
        statWriter.writerow([fitness[i], avg_scores[i]])

    csvfile.close()


def save_species_count(species_log, view=False, filename='speciation.csv'):
    """ Visualizes speciation throughout evolution. """
    csvfile = open(filename, 'wb')
    statWriter = csv.writer(csvfile, delimiter=' ')


    num_generations = len(species_log)
    num_species = max(map(len, species_log))
    curves = []
    for gen in species_log:
        species = [0] * num_species
        species[:len(gen)] = gen
        statWriter.writerow(species)

    csvfile.close()

def save_species_fitness(species_fitness_log, view=False, filename='species_fitness.csv'):
    """ Visualizes speciation throughout evolution. """
    csvfile = open(filename, 'wb')
    statWriter = csv.writer(csvfile, delimiter=' ')
    num_generations = len(species_fitness_log)
    num_species = max(map(len, species_fitness_log))
    curves = []
    for gen in species_fitness_log:
        species = ["NA"] * num_species
        species[:len(gen)] = gen
        statWriter.writerow(species)

    csvfile.close()
