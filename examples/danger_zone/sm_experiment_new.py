"""
Simple example using the tile structure creation task.
"""

import os

import neat
import gym

from examples.danger_zone.DangerZoneExperiment import DangerZoneExperiment, WorstChangeKeeperExperimentRunner
from examples.experiment_functions import StateMachineController, ExperimentRunner

# Important variables.
from neat.reproduction_state_machine import ReproductionStateMachineOnly, StateSeparatedSpeciesSet
from neat.stagnation import MarkAllStagnation
from neat.state_machine_full_genome import StateMachineFullGenome

experiment_name = 'SM_state_dependent'
num_steps = 150
num_generations = 100
num_runs = 5
num_trails = 1
config_name = 'config-sm_state_species'

if __name__ == '__main__':

    env = gym.make('danger-zone-v0')
    runner = ExperimentRunner(env, num_steps, StateMachineController)

    # Create learning configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = neat.Config(StateMachineFullGenome,
                         ReproductionStateMachineOnly,
                         StateSeparatedSpeciesSet,
                         MarkAllStagnation,
                         config_path)

    # Create and run experiment.
    experiment = DangerZoneExperiment(config, runner, num_generations, experiment_name, num_trails)

    for i in range(num_runs):
        experiment.run(experiment_name + str(i))
