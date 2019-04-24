import os
import time
import neat
from gym_multi_robot import visualize
from gym_multi_robot.object_serializer import ObjectSerializer


class SingleExperiment:
    """ This class gives the functions required to run a single experiment."""

    def __init__(self, learning_config, exp_runner, num_generations, exp_name='', num_trails=1, base_directory=''):
        self.exp_name = exp_name
        self.learning_config = learning_config
        self.exp_runner = exp_runner
        self.num_generations = num_generations
        self.num_trails = num_trails
        self.winner = None  # Stores the winner of the last experiment.
        self.stats = None   # Stores the stats about the last experiment.
        self.base_directory = base_directory

    def eval_genomes(self, genomes, config):
        start_time = time.time()

        for genome_id, genome in genomes:

            self.process_genome(genome, config)
            # sub rewards.

        end_time = time.time()
        time_diff = end_time - start_time
        avg_time = time_diff / len(genomes)

        print("generation total_runtime: %s seconds, avg_runtime: %s seconds" % (time_diff, avg_time))

    def process_genome(self, genome, config):
        """ This function processes a genome to finds its fitness and possibly other details. """
        genome.fitness = self.exp_runner.run_multiple_trails(genome, config, self.num_trails)

    def run(self, name=None):
        """ Runs the experiment.
        Name parameter can be used to update the name of the experiment.
        """
        if name is not None:
            self.exp_name = name

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(self.learning_config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        p.add_reporter(self.stats)

        # Run experiments
        self.winner = p.run(self.eval_genomes, self.num_generations)

        # Output results and statistics.
        self.output_stats()
        self.output_winner()

    def output_winner(self):
        """This function outputs the current winner in graph and in pickle file."""
        self.init_base_directory()

        net_filename = self.base_directory + 'graph_winner' + str(self.exp_name)
        genome_filename = self.base_directory + 'winner' + str(self.exp_name)

        self.exp_runner.draw(self.winner, self.learning_config, net_filename)

        ObjectSerializer.serialize(self.winner, genome_filename)

        print(self.winner)

    def output_stats(self):
        """ This function outputs the statistics in figures and in reusable objects."""
        self.init_base_directory()

        fitness_out_file = self.base_directory + 'avg_fitness_' + str(self.exp_name) + '.svg'
        species_out_file = self.base_directory + 'species_' + str(self.exp_name) + '.svg'
        stats_out_file = self.base_directory + 'stats' + str(self.exp_name)

        visualize.visualize_stats(self.stats, fitness_out_file, species_out_file)
        ObjectSerializer.serialize(self.stats, stats_out_file)

    def init_base_directory(self):
        """ This function checks whether the base directory exists and creates it if it doesn't. """

        if self.base_directory != '' and not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)
