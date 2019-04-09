"""
Simple example using the tile structure creation task.
"""

import os

import neat
import gym

from examples.experiment_functions import NEATExperimentRunner, SMExperimentRunner
from examples.experiment_template import SingleExperiment

# Important variables.
from neat.state_machine_genome import StateMachineGenome

experiment_name = 'SM'
num_steps = 150
num_generations = 100
num_runs = 1
num_trails = 1
config_name = 'config-sm_2_state'

if __name__ == '__main__':

    env = gym.make('danger-zone-v0')
    runner = SMExperimentRunner(env, num_steps)

    # Create learning configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = neat.Config(StateMachineGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create and run experiment.
    experiment = SingleExperiment(config, runner, num_generations, experiment_name, num_trails)

    for i in range(num_runs):
        experiment.run(experiment_name + str(i))
