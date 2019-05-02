"""
Simple example using the tile structure creation task.
"""

import os

import neat
import gym

from examples.danger_zone.DangerZoneExperiment import DangerZoneExperiment, WorstChangeKeeperExperimentRunner
from examples.experiment_functions import StateMachineController, ExperimentRunner

# Important variables.
from neat.reproduction_mutation_only import ReproductionMutationOnly
from neat.state_machine_genome import StateMachineGenome

experiment_name = 'SM_free_redo'
num_steps = 150
num_generations = 100
num_runs = 5
num_trails = 1
config_name = 'config-sm_free_states'

if __name__ == '__main__':

    env = gym.make('danger-zone-v0')
    runner = ExperimentRunner(env, num_steps, StateMachineController)

    # Create learning configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = neat.Config(StateMachineGenome,
                         ReproductionMutationOnly,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_path)

    # Create and run experiment.
    experiment = DangerZoneExperiment(config, runner, num_generations, experiment_name, num_trails, 'results/SM_free/')

    for i in range(num_runs):
        experiment.run(experiment_name + str(i))
