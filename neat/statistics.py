from __future__ import print_function

import csv

from neat.math_util import mean


def get_species_sizes(population):
    all_species = set()
    for gen_data in population.generation_statistics:
        all_species = all_species.union(gen_data.keys())

    max_species = max(all_species)
    species_counts = []
    for gen_data in population.generation_statistics:
        species = [len(gen_data.get(sid, [])) for sid in range(1, max_species + 1)]
        species_counts.append(species)

    return species_counts


def get_species_fitness(population, null_value=''):
    all_species = set()
    for gen_data in population.generation_statistics:
        all_species = all_species.union(gen_data.keys())

    max_species = max(all_species)
    species_fitness = []
    for gen_data in population.generation_statistics:
        member_fitness = [gen_data.get(sid, []) for sid in range(1, max_species + 1)]
        fitness = []
        for mf in member_fitness:
            if mf:
                fitness.append(mean(mf))
            else:
                fitness.append(null_value)
        species_fitness.append(fitness)

    return species_fitness


def save_stats(statistics, delimiter=' ', filename='fitness_history.csv'):
    """ Saves the population's best and average fitness. """
    with open(filename, 'w') as f:
        w = csv.writer(f, delimiter=delimiter)

        best_fitness = [c.fitness for c in statistics.most_fit_genomes]
        avg_fitness = statistics.get_average_fitness()
        for best, avg in zip(best_fitness, avg_fitness):
            w.writerow([best, avg])


def save_species_count(statistics, delimiter=' ', filename='speciation.csv'):
    """ Log speciation throughout evolution. """
    with open(filename, 'w') as f:
        w = csv.writer(f, delimiter=delimiter)
        for s in get_species_sizes(statistics):
            w.writerow(s)


def save_species_fitness(statistics, delimiter=' ', null_value='NA', filename='species_fitness.csv'):
    """ Log species' average fitness throughout evolution. """
    with open(filename, 'w') as f:
        w = csv.writer(f, delimiter=delimiter)
        for s in get_species_fitness(statistics, null_value):
            w.writerow(s)
