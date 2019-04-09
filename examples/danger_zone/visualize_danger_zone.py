import os
import pickle

import gym
import gym_multi_robot

from examples.experiment_functions import SMExperimentRunner
from neat import Config, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, DefaultGenome
from neat.state_machine_genome import StateMachineGenome
import matplotlib.pyplot as plt


class DangerZoneVisualizer(SMExperimentRunner):

    def __init__(self, gym_environment, num_steps, render):
        super().__init__(gym_environment, num_steps, render)

        self.x_data = []
        self.y_data = []

    def check_render(self, time_step=0):
        # Make a graph containing the path of the robot.

        if self.render:

            self.x_data.append(time_step)
            self.y_data.append(self.env.robot_location)

    def produce_image(self):

        plt.plot(self.x_data, self.y_data)
        plt.xlabel('Timesteps')
        plt.ylabel('Location')
        plt.hlines(self.env.area_size - self.env.danger_size, min(self.x_data), max(self.x_data), 'r')
        plt.savefig('trajectory.png')


if __name__ == '__main__':

    genome_path = 'winnerSM0.pickle'
    config_path = 'config-sm_2_state'
    env_name = 'danger-zone-v0'
    num_steps = 200
    delay_time = 50

    local_dir = os.path.dirname(__file__)

    # First load the genome
    genome = pickle.load(open(genome_path, "rb"))

    print('obtained fitness is: ' + str(genome.fitness))
    print(genome)

    config = Config(StateMachineGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, config_path)
    print(config.genome_config)
    vis = DangerZoneVisualizer(gym.make(env_name), num_steps, True)
    vis.run(genome, config)
    vis.produce_image()
