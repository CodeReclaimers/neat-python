"""
Simple example using the tile structure creation task.
"""

import os

import neat
import gym

from examples.robot_experiments import gym_robot_experiment
from examples.experiment_functions import ExperimentRunner, StateMachineController

# Important variables.
from neat.state_machine_genome_partially_fixed import StateMachineGenomeFixed

experiment_name = 'SM_fixed_layout'
num_steps = 150
num_generations = 100
num_runs = 1
num_trails = 1
config_name = 'config-sm_fixed'

if __name__ == '__main__':

    env = gym.make('danger-zone-v0')
    runner = ExperimentRunner(env, num_steps, StateMachineController)

    # Create learning configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = neat.Config(StateMachineGenomeFixed,
                         neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Create and run experiment.
    experiment = gym_robot_experiment(config, runner, num_generations, experiment_name, num_trails,
                                      'results/SM_fixed_layout/')

    for i in range(num_runs):
        experiment.run(experiment_name + str(i))
