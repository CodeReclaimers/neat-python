import pygame

from neat import nn
from neat.state_machine_network import StateMachineNetwork


class GenomeVisualisor:

    def __init__(self, env, genome, config, delay_time):
        self.env = env
        self.genome = genome
        self.delay_time = delay_time
        self.config = config

    def run(self, time_steps):
        """ This function should run and visualise the given genome, for the given number of timesteps."""
        pass


class StateMachineVisualisor(GenomeVisualisor):

    def run(self, time_steps):
        # Run the genome
        net = StateMachineNetwork.create(self.genome, self.config.genome_config)
        observation = self.env.reset()

        state = 0
        for _ in range(time_steps):
            state, action = net.activate(state,observation)
            observation, _, _, _ = self.env.step(action)
            self.env.render()
            pygame.time.wait(self.delay_time)


class StateMachineSwarmVisualisor(GenomeVisualisor):
    """ This class visualizes a state machine genome in the given environment"""

    def run(self, time_steps):
        # Run the genome
        net = StateMachineNetwork.create(self.genome, self.config.genome_config)
        observation = self.env.reset()

        states = [0 for _ in range(len(observation))]
        for _ in range(time_steps):
            output = [net.activate(states[i], observation[i]) for i in range(len(observation))]
            states = [state for state, _ in output]
            actions = [action for _, action in output]
            observation, _, _, _ = self.env.step(actions)
            self.env.render()
            pygame.time.wait(self.delay_time)


class NeatSwarmVisualisor(GenomeVisualisor):
    """ Visualizes a NEAT genome in an environment that has multiple robots (a swarm)."""

    def run(self, time_steps):
        # Run the genome
        net = nn.FeedForwardNetwork.create(self.genome, self.config)
        observation = self.env.reset()

        for _ in range(time_steps):
            actions = [net.activate(observation[i]) for i in range(len(observation))]
            observation, _, _, _ = self.env.step(actions)
            self.env.render()
            pygame.time.wait(self.delay_time)


class NEATVisualisor(GenomeVisualisor):
    """ This class visualizes a NEAT genome in an environment with of a single robot. """

    def run(self, time_steps):
        # Run the genome
        net = nn.FeedForwardNetwork.create(self.genome, self.config)
        observation = self.env.reset()

        for _ in range(time_steps):
            actions = net.activate(observation)
            observation, _, _, _ = self.env.step(actions)
            self.env.render()
            pygame.time.wait(self.delay_time)

