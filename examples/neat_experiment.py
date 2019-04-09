import os
import sys

import gym

from examples.experiment_functions import NEATSwarmExperimentRunner
from examples.experiment_template import SingleExperiment

# This script describes an state machine experiment where all the values are specified in a config file.
# Config file can and should be given as a parameter.
from neat import Config, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, DefaultGenome
from neat.state_machine_genome import StateMachineGenome

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('One argument expected, which is the path to the config file.')
    else:
        config_name = sys.argv[1]

        # Create learning configuration.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_name)
        config = Config(DefaultGenome, DefaultReproduction,
                        DefaultSpeciesSet, DefaultStagnation,
                        config_path)

        env = gym.make(config.env_name)
        runner = NEATSwarmExperimentRunner(env, config.num_steps)

        # Create and run experiment.
        experiment = SingleExperiment(config, runner, config.num_generations, config.experiment_name)

        for i in range(config.num_runs):
            experiment.run(config.experiment_name + str(i))
