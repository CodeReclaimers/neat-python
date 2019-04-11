"""
Simple example using the tile structure creation task.
"""

import os

import numpy as np

import neat
import gym

from examples.danger_zone.DangerZoneExperiment import DangerZoneExperiment
from examples.experiment_functions import ExperimentRunner, FeedForwardNetworkController
from examples.experiment_template import SingleExperiment

# Important variables.
experiment_name = 'NEAT'
num_steps = 150
num_generations = 100
num_runs = 1
num_trials = 1
config_name = 'config-feedforward'


if __name__ == '__main__':

    env = gym.make('danger-zone-v0')
    runner = ExperimentRunner(env, num_steps, FeedForwardNetworkController)

    # Create learning configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create and run experiment.
    experiment = DangerZoneExperiment(config, runner, num_generations, experiment_name, num_trials)

    for i in range(num_runs):
        experiment.run(experiment_name + str(i))
