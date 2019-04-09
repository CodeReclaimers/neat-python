from neat.reporting import BaseReporter
import matplotlib.pyplot as plt


class FitnessSpreadReporter(BaseReporter):
    """" This reporter creates a visualization of how the fitness for each generation is spread."""

    def __init__(self, experiment_name):
        self.output_name = experiment_name + '_fitness_'
        self.generation = 0

    def end_generation(self, config, population, species_set):

        # the histogram of the data
        fitness = [0 if genome.fitness is None else genome.fitness for genome in population.values()]

        n, bins, patches = plt.hist(fitness, 20, facecolor='g', alpha=0.75)
        plt.xlabel('Fitness')
        plt.ylabel('Genomes in bin')
        plt.title('Fitness division of generation ' + str(self.generation))
        plt.grid(True)

        plt.savefig(self.output_name + str(self.generation) + '.png', bbox_inches='tight')
        plt.clf()

        self.generation += 1

        print('--- Outputted Fitness division---')
