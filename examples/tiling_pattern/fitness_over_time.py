import csv
import os
import pickle
import matplotlib.pyplot as plt

import gym

from examples.experiment_functions import SMSwarmExperimentRunner
from neat import Config, DefaultSpeciesSet, DefaultStagnation, DefaultReproduction
from neat.state_machine_genome import StateMachineGenome
from neat.state_machine_network import StateMachineNetwork

genome_path = 'winnerSM_4_reference_block_long0.pickle'
config_name = 'config-state_machine'
env_name = 'tiling-pattern11x11-block-v0'
output_file = 'fitness.csv'
num_steps = 3000

if __name__ == '__main__':

    # First load the genome
    genome = pickle.load(open(genome_path, "rb"))

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_name)
    config = Config(StateMachineGenome, DefaultReproduction,
                    DefaultSpeciesSet, DefaultStagnation, config_path)

    # Create the environment
    env = gym.make(env_name)

    # Run the genome
    net = StateMachineNetwork.create(genome, config.genome_config)
    observation = env.reset()
    fitnesses = []

    states = [0 for _ in range(len(observation))]
    for i in range(num_steps):
        output = [net.activate(states[i], observation[i]) for i in range(len(observation))]
        states = [state for state, _ in output]
        actions = [action for _, action in output]
        observation, _, _, _ = env.step(actions)
        fitnesses.append(env.get_fitness())

    # Show plot of fitness over time.
    x = [i for i in range(num_steps)]
    plt.xlabel('Time step')
    plt.ylabel('Fitness')
    plt.plot(x, fitnesses)
    plt.show()
