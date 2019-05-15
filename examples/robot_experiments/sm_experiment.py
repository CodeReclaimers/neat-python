"""
Simple example using the tile structure creation task.
"""

import os
import sys

import neat
import gym

from examples.experiment_functions import StateMachineController, ExperimentRunner

# Important variables.
from examples.robot_experiments.gym_robot_experiment import GymRobotExperiment
from neat.reproduction_mutation_only import ReproductionMutationOnly
from neat.state_machine_genome import StateMachineGenome

experiment_name = 'SM_free'
num_steps = 300
num_generations = 100
num_runs = 5
num_trails = 5

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Expected 2 argument: the gym environment to use and the place where the config file can be found and the"
              "resulst should be stored.")
        print("python neat_experiment.py gym_env config_file results_directory/")
        exit()

    env_name = sys.argv[1]
    config_location = sys.argv[2] + 'config-sm_free_states'
    results_dir = sys.argv[2] + 'results/SM_free_rand/'

    env = gym.make(env_name)
    runner = ExperimentRunner(env, num_steps, StateMachineController)

    # Create learning configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_location)
    config = neat.Config(StateMachineGenome,
                         ReproductionMutationOnly,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_path)

    # Create and run experiment.
    experiment = GymRobotExperiment(config, runner, num_generations, experiment_name, num_trails, results_dir)

    for i in range(num_runs):
        experiment.run(experiment_name + str(i))

