import glob
import os
import pickle

import matplotlib.pyplot as plt

import numpy as np


def get_avg_fitness_of_dir(files_location):
    """ This function extracts the average and best fitness of multiple run and returns an average over generations. """

    stat_files = glob.glob(files_location + 'stats*.pickle')

    mean_fitnesses = []
    best_fitnesses = []

    for file in stat_files:
        stats = pickle.load(open(file, 'rb'))

        mean_fitness = stats.get_fitness_mean()
        best_fitness = [c.fitness for c in stats.most_fit_genomes]

        mean_fitnesses.append(mean_fitness)
        best_fitnesses.append(best_fitness)

    summed_best_fitness = sum(np.array(best_fitnesses))
    summed_mean_fitness = sum(np.array(mean_fitnesses))

    avg_best_fitness = summed_best_fitness / len(stat_files)
    avg_mean_fitness = summed_mean_fitness / len(stat_files)

    return avg_mean_fitness, avg_best_fitness


if __name__ == '__main__':

    neat_avg_mean_fitness, neat_avg_best_fitness = get_avg_fitness_of_dir('NEAT/')
    free_avg_mean_fitness, free_avg_best_fitness = get_avg_fitness_of_dir('SM_free/')
    depend_avg_mean_fitness, depend_avg_best_fitness = get_avg_fitness_of_dir('SM_state_dependent/')

    gen_count = list(range(1, len(neat_avg_mean_fitness) + 1))

    plt.plot(gen_count, neat_avg_mean_fitness, label='NEAT average')
    plt.plot(gen_count, neat_avg_best_fitness, label='NEAT best')
    plt.plot(gen_count, free_avg_mean_fitness, label='SM Free Evolvement average')
    plt.plot(gen_count, free_avg_best_fitness, label='SM Free Evolvement best')
    plt.plot(gen_count, depend_avg_mean_fitness, label='SM State Dependent average')
    plt.plot(gen_count, depend_avg_best_fitness, label='SM State Dependent average')
    plt.legend(loc='lower right')

    plt.show()


