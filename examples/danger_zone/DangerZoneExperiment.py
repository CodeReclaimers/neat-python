import math

from examples.experiment_functions import ExperimentRunner
from examples.experiment_template import SingleExperiment


class DangerZoneExperiment(SingleExperiment):
    """ This class adds the custom functions that need to be applied in a danger zone experiment.
        In this case this is the plotting of the trajectory.
    """

    def output_winner(self):
        super().output_winner()

        self.exp_runner.render = True

        self.exp_runner.run(self.winner, self.learning_config)
        self.exp_runner.env.produce_trajectory(self.base_directory + 'trajectory' + str(self.exp_name) + '.png')

        self.exp_runner.render = False

    def process_genome(self, genome, config):

        genome.fitness = self.exp_runner.run_multiple_trails(genome, config, self.num_trails)


class WorstChangeKeeperExperimentRunner(ExperimentRunner):
    """
    This experiment runner keeps the worst difference in observation, such that the robot can create a state transition
    based on this information.
    The goal of this is to provide better guidance to the process of selecting right transitions.
    """

    def __init__(self, gym_environment, num_steps, controller_class):
        super().__init__(gym_environment, num_steps, controller_class)

        print("Initialised this one.")
        self.worst_observation_pair = None

    def get_worst_observation_pair(self):
        return self.worst_observation_pair

    def run(self, genome, config):
        """ This function should run the experiment with the given genome and configuration.
            Should be implemented by the subclasses with an implementation of how to run.
            Returned should be the fitness of swarm behaviour, as indicated by the environment.
        """
        controller = self.controller_class()
        controller.reset(genome, config)
        observation = self.env.reset()
        fitness = 0

        # Additional parameters that can be used for when a bad decision is taken.
        worst_change = math.inf
        self.worst_observation_pair = None

        previous_fitness = None
        previous_observation = observation

        for i in range(self.num_steps):

            output = controller.step(observation)
            observation, fitness, done, _ = self.env.step(output)

            if previous_fitness is None:
                previous_fitness = fitness
            else:
                new_fitness = fitness - previous_fitness
                fitness_difference = previous_fitness - new_fitness

                # If the difference in fitness now and in the previous round is worse than encountered, update.
                if fitness_difference < worst_change:
                    self.worst_observation_pair = (previous_observation, observation)
                    worst_change = fitness_difference

                previous_fitness = new_fitness
                previous_observation = observation

            self.check_render()
            if done:
                break

        return fitness
