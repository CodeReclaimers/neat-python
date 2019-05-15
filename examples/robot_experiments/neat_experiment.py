"""
Simple example using the tile structure creation task.
"""

import os
import sys

import neat
import gym

from examples.experiment_functions import ExperimentRunner, FeedForwardNetworkController

# Important variables.
from examples.robot_experiments.gym_robot_experiment import GymRobotExperiment

experiment_name = 'NEAT'
num_steps = 300
num_generations = 100
num_runs = 5
num_trials = 5


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Expected 2 argument: the gym environment to use and the place where the config file can be found and the"
              "resulst should be stored.")
        print("python neat_experiment.py gym_env config_file results_directory/")
        print(sys.argv)
        exit()

    env_name = sys.argv[1]
    config_location = sys.argv[2] + 'config-feedforward'
    results_dir = sys.argv[2] + 'results/NEAT_rand/'

    env = gym.make(env_name)
    runner = ExperimentRunner(env, num_steps, FeedForwardNetworkController)

    # Create learning configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_location)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create and run experiment.
    experiment = GymRobotExperiment(config, runner, num_generations, experiment_name, num_trials, results_dir)

    for i in range(num_runs):
        experiment.run(experiment_name + str(i))
