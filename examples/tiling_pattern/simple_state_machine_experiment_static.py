"""
Simple example using the tile structure creation task.
"""

from __future__ import print_function
import os
import time

import neat
import gym
import gym_multi_robot
from gym_multi_robot import visualize

from examples.tiling_pattern.tiling_pattern_functions import output_winner
from neat import DefaultSpeciesSet, DefaultReproduction, DefaultStagnation
from neat.state_machine_genome import StateMachineGenome
from neat.state_machine_network import StateMachineNetwork

num_steps = 3000
num_generations = 100

env = gym.make('tiling-pattern7x5-static-v0')


def eval_genomes(genomes, config):
    count = 0
    for genome_id, genome in genomes:
        count += 1
        net = StateMachineNetwork.create(genome, config.genome_config)
        start_time = time.time()
        genome.fitness = run_environment(net)
        # sub rewards.
        end_time = time.time()
        avg_time = end_time - start_time

        print("%d : avg_runtime: %s seconds ---" %(count, avg_time))


def run_environment(net):
    observation = env.reset()

    states = [0 for _ in range(len(observation))]
    for i in range(num_steps):
        output = [net.activate(states[i], observation[i]) for i in range(len(observation))]
        states = [state for state, _ in output]
        actions = [action for _, action in output]
        observation, _, _, _ = env.step(actions)

    reward = env.get_fitness()

    return reward


def run(config_file):
    # Load configuration.
    config = neat.Config(StateMachineGenome, DefaultReproduction,
                         DefaultSpeciesSet, DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, num_generations)

    visualize.visualize_stats(stats, fitness_out_file='sm_avg_fitness_static.svg',
                              species_out_file='sm_species_static.svg')
    output_winner(winner, config, net_filename='sm_winner_static', genome_filename='sm_winner_static')


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-state_machine')
    run(config_path)