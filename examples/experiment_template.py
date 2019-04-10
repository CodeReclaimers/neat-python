import time
import neat
from gym_multi_robot import visualize
from gym_multi_robot.object_serializer import ObjectSerializer


class SingleExperiment:
    """ This class gives the functions required to run a single experiment."""

    def __init__(self, learning_config, exp_runner, num_generations, exp_name='', num_trails=1):
        self.exp_name = exp_name
        self.learning_config = learning_config
        self.exp_runner = exp_runner
        self.num_generations = num_generations
        self.num_trails = num_trails
        self.winner = None  # Stores the winner of the last experiment.
        self.stats = None   # Stores the stats about the last experiment.

    def eval_genomes(self, genomes, config):
        start_time = time.time()

        for genome_id, genome in genomes:
            genome.fitness = self.exp_runner.run_multiple_trails(genome, config, self.num_trails)
            # sub rewards.

        end_time = time.time()
        time_diff = end_time - start_time
        avg_time = time_diff / len(genomes)

        print("generation total_runtime: %s seconds, avg_runtime: %s seconds" % (time_diff, avg_time))

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

        net_filename = 'graph_winner' + str(self.exp_name)
        genome_filename = 'winner' + str(self.exp_name)

        node_names = {-1: 'hold', -2: 'on object', -3: '1_obstacle', -4: '1_tile', -5: '1_robot', -6: '2_obstacle',
                      -7: '2_tile', -8: '2_robot', -9: '3_obstacle', -10: '3_tile', -11: '3_robot', -12: '4_obstacle',
                      -13: '4_tile', -14: '4_robot', -15: '5_obstacle', -16: '5_tile', -17: '5_robot',
                      0: 'drive', 1: 'rotation', 2: 'pickup', 3: 'put down'}
        self.exp_runner.draw(self.learning_config, self.winner, node_names=node_names, filename=net_filename)

        ObjectSerializer.serialize(self.winner, genome_filename)

        print(self.winner)

    def output_stats(self):
        """ This function outputs the statistics in figures and in reusable objects."""
        fitness_out_file = 'avg_fitness_' + str(self.exp_name) + '.svg'
        species_out_file = 'species_' + str(self.exp_name) + '.svg'
        stats_out_file = 'stats' + str(self.exp_name)

        visualize.visualize_stats(self.stats, fitness_out_file,species_out_file)
        ObjectSerializer.serialize(self.stats, stats_out_file)
