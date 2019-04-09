import os
import pickle

import gym
import gym_multi_robot

from examples.game_visualizor import StateMachineSwarmVisualisor
from neat import Config, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation
from neat.state_machine_genome import StateMachineGenome

if __name__ == '__main__':

    general_path = '/experiment_results/tiling_pattern_task/state_machine/states_experiment/4/'
    genome_name = 'winnersm_4_state4.pickle'
    config_name = 'config-sm_4_state'
    env_name = 'tiling-pattern11x11-block-v0'
    num_steps = 2000
    delay_time = 50

    local_dir = os.path.dirname(__file__)

    genome_path = local_dir + general_path + genome_name
    config_path = local_dir + general_path + config_name

    # First load the genome
    genome = pickle.load(open(genome_path, "rb"))

    print('obtained fitness is: ' + str(genome.fitness))

    config = Config(StateMachineGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, config_path)
    vis = StateMachineSwarmVisualisor(gym.make(env_name), genome, config, delay_time)

    vis.run(num_steps)
