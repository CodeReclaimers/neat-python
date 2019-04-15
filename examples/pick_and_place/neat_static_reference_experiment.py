"""
Simple example using the tile structure creation task.
"""

import os

import numpy as np

import neat
import gym

from examples.experiment_functions import NEATExperimentRunner
from examples.experiment_template import SingleExperiment

# Important variables.
experiment_name = 'NEAT_reference_static'
num_steps = 3000
num_robots = 5
num_generations = 100
num_runs = 1
num_trials = 5
config_name = 'config-feedforward'


class NEATPickAndPlaceExperimentRunner(NEATExperimentRunner):
    """ This class runs an gym experiment with a single agent, as opposed to multiple agents."""

    def run(self, genome, config):
        """ This function runs an experiment for a NEAT genome, given a genome and the required variables."""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation = self.env.reset()

        for i in range(self.num_steps):

            output = net.activate(observation['observation'])
            observation, _, done, _ = self.env.step(output)

            if self.render:
                self.env.render()

            if done:
                break

        rel_pos_arm = observation['observation'][6:9]       # Position from arm to object
        rel_obj_to_goal = observation['observation'][3:6] - observation['desired_goal']  # Position from object to goal.
        fitness = np.linalg.norm(rel_pos_arm) + rel_obj_to_goal

        return 5 - fitness


if __name__ == '__main__':

    env = gym.make('FetchPickAndPlace-v1')
    runner = NEATPickAndPlaceExperimentRunner(env, num_steps)

    # Create learning configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create and run experiment.
    experiment = SingleExperiment(config, runner, num_generations, experiment_name, num_trials)

    for i in range(num_runs):
        experiment.run(experiment_name + str(i))
