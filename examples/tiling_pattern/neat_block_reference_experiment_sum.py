"""
Simple example using the tile structure creation task.
"""

import os

import neat
import gym

from examples.experiment_functions import NEATSwarmExperimentRunner
from examples.experiment_template import SingleExperiment

# Important variables.
experiment_name = 'NEAT_reference_block_sum'
num_steps = 3000
num_generations = 100
num_runs = 5
config_name = 'config-feedforward'

gym.register(
    id='tiling-pattern11x11-block-sum-v0',
    entry_point='gym_multi_robot.envs:SummedTilingPatternEnv',
    kwargs={'env_storage_path': 'tiles11x11_block.pickle'}
)

if __name__ == '__main__':
    """ Experiment where fitness is gotten by summing over all fitnesses at every timestep (with equal weigth).
    """

    env = gym.make('tiling-pattern11x11-block-sum-v0')
    runner = NEATSwarmExperimentRunner(env, num_steps)

    # Create learning configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # change fitness threshold
    config.fitness_threshold = 3000 * 100

    # Create and run experiment.
    experiment = SingleExperiment(config, runner, num_generations, experiment_name)

    for i in range(num_runs):
        experiment.run(experiment_name + str(i))
