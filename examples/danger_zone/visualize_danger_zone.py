import os
import pickle

import gym
import gym_multi_robot

from examples.experiment_functions import ExperimentRunner, StateMachineController
from neat import Config, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation
from neat.state_machine_genome import StateMachineGenome

if __name__ == '__main__':

    genome_path = 'winnerSM0.pickle'
    config_path = 'config-sm_2_state'
    env_name = 'danger-zone-v0'
    num_steps = 200
    delay_time = 50

    config = Config(StateMachineGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, config_path)
    local_dir = os.path.dirname(__file__)

    # First load the genome
    genome = pickle.load(open(genome_path, "rb"))

    print('obtained fitness is: ' + str(genome.fitness))

    env = gym.make(env_name)

    vis = ExperimentRunner(env, num_steps, StateMachineController, True)
    vis.run(genome, config)
    env.produce_trajectory('trajectory.png')
