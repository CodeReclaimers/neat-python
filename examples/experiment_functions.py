from gym_multi_robot import visualize

import neat
from neat.state_machine_network import StateMachineNetwork


class ExperimentRunner:
    """ This class represents an experiment runner, so an instance which can be used to run gym experiments.
        Note that the fitness is only requested using the get_fitness() function of the environment after the last step.
        So make sure the gym environment provides this.
    """

    def __init__(self, gym_environment, num_steps, render=False):
        self.env = gym_environment
        self.num_steps = num_steps
        self.render = render

    def run_multiple_trails(self, genome, config, num_trails):
        """ This function runs multiple trials with the same genome and environment.
            This can be useful when the environments contains randomness, since multiple situations are evaluated.
            returns the winner and the stats for each of the trials.
        """
        reward = 0
        for _ in range(num_trails):
            reward += self.run(genome, config)

        return reward / num_trails

    def run(self, genome, config):
        """ This function should run the experiment with the given genome and configuration.
            Should be implemented by the subclasses with an implementation of how to run.
            Returned should be the fitness of swarm behaviour, as indicated by the environment.
        """
        pass

    def draw(self, genome, config):
        """ This function should draw the given genome. It depends on the genome that is actually used."""
        pass

    def check_render(self, time_step=0):
        if self.render:
            self.env.render()


class NEATExperimentRunner(ExperimentRunner):
    """ This class is an abstract class for NEAT experiments. run() function has to be implemented."""

    def draw(self, genome, config, node_names=None, filename=None):
        visualize.draw_net(genome, config, node_names=node_names, filename=filename)

    def run(self, genome, config):

        """ This function runs an experiment for a NEAT genome, given a genome and the required variables."""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation = self.env.reset()
        fitness = 0

        for i in range(self.num_steps):

            output = net.activate(observation)
            observation, fitness, done, _ = self.env.step(output)

            self.check_render(i)
            if done:
                break

        return fitness


class NEATSwarmExperimentRunner(NEATExperimentRunner):
    """ This class can be used to run Neat experiments."""

    def run(self, genome, config):
        """ This function runs an experiment for a NEAT genome, given a genome and the required variables."""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation = self.env.reset()

        for i in range(self.num_steps):

            output = [net.activate(observation[i]) for i in range(len(observation))]
            observation, _, done, _ = self.env.step(output)

            self.check_render(i)
            if done:
                print('Done after ' + str(i) + ' steps')
                break

        return self.env.get_fitness()


class SMExperimentRunner(ExperimentRunner):
    """ This function runs a state machine experiment with a single robot. """

    def draw(self, genome, config, node_names=None, filename=None):
        visualize.draw_state_machine(genome, config, node_names=node_names, filename=filename)

    def run(self, genome, config):
        """ This function runs an experiment for a SM genome, given a genome and the required variables."""
        net = StateMachineNetwork.create(genome, config.genome_config)
        observation = self.env.reset()

        fitness = 0
        state = 0

        for i in range(self.num_steps):
            state, action = net.activate(state, observation)
            observation, fitness, done, _ = self.env.step(action)

            self.check_render(i)
            if done:
                break

        return fitness


class SMSwarmExperimentRunner(SMExperimentRunner):
    """ This class can be used to run state machine experiments."""

    def run(self, genome, config):
        """ This function runs an experiment for a SM genome, given a genome and the required variables."""
        net = StateMachineNetwork.create(genome, config.genome_config)
        observation = self.env.reset()

        states = [0 for _ in range(len(observation))]
        for i in range(self.num_steps):
            output = [net.activate(states[i], observation[i]) for i in range(len(observation))]
            states = [state for state, _ in output]
            actions = [action for _, action in output]
            observation, _, done, _ = self.env.step(actions)

            self.check_render(i)
            if done:
                break

        return self.env.get_fitness()
