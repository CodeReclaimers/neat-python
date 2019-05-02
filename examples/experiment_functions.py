from subprocess import CalledProcessError

from gym_multi_robot import visualize

import neat
from neat.state_machine_network import StateMachineNetwork


class ExperimentRunner:
    """ This class represents an experiment runner, so an instance which can be used to run gym experiments.
        Note that the fitness is only requested using the get_fitness() function of the environment after the last step.
        So make sure the gym environment provides this.
    """

    def __init__(self, gym_environment, num_steps, controller_class, render=False):
        self.env = gym_environment
        self.num_steps = num_steps
        self.controller_class = controller_class
        self.render = render
        self.controller = None

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
        self.controller = self.controller_class()
        self.controller.reset(genome, config)
        observation = self.env.reset()
        fitness = 0

        for i in range(self.num_steps):

            output = self.controller.step(observation)
            observation, fitness, done, _ = self.env.step(output)

            self.check_render()
            if done:
                break

        return fitness

    def draw(self, genome, config, file_name='winner.svg'):
        """ This function should draw the given genome. It depends on the genome that is actually used."""
        self.controller_class.draw(genome, config, file_name)

    def check_render(self):
        if self.render:
            self.env.render()


class SwarmExperimentRunner(ExperimentRunner):
    """ This class describes an experiment runner for a swarm, ie. multiple robots with the same controller."""

    def run(self, genome, config):

        observations = self.env.reset()

        # Spawn as many controllers as there are observations.
        self.controller = [self.controller_class() for _ in range(len(observations))]

        # Reset all controllers with the genome and config.
        for controller in self.controller:
            controller.reset(genome, config)

        for i in range(self.num_steps):

            output = [self.controller[i].step(observations[i]) for i in range(len(observations))]
            observations, _, done, _ = self.env.step(output)

            self.check_render()
            if done:
                break

        return self.env.get_fitness()


class SimulationController:
    """ This class calculates the actions in the simulation using the given control mechanism."""

    def __init__(self):
        self.net = None

    def reset(self, genome, config):
        """ This function resets the stepper indicating that a new simulation is started"""
        pass

    def step(self, observation):
        """ This function calculates the desired course of action based on the given observation."""
        pass

    @classmethod
    def draw(cls, genome, config, file_name):
        """ This function draws the given controller."""
        pass


class FeedForwardNetworkController(SimulationController):
    """ This class calculates the next actions based on a feed forward network."""

    def reset(self, genome, config):
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def step(self, observation):
        return self.net.activate(observation)

    @classmethod
    def draw(cls, genome, config, file_name):
        visualize.draw_net(config, genome, filename=file_name)


class StateMachineController(SimulationController):
    """ This class calculates the next actions based on a state machine. The difference here is that the current state,
        needs to be taken into account and updated, as this is also part of the state machine.
    """

    def __init__(self):
        SimulationController.__init__(self)
        self.current_state = 0

    def reset(self, genome, config):
        self.net = StateMachineNetwork.create(genome, config.genome_config)
        self.current_state = 0

    def step(self, observation):
        new_state, actions = self.net.activate(self.current_state, observation)
        self.current_state = new_state

        return actions

    @classmethod
    def draw(cls, genome, config, file_name):
        try:
            visualize.draw_state_machine(config, genome, filename=file_name)
        except CalledProcessError:
            print('State machine graph failed, continuing without producing graph.')


class LoggingStateMachineController(StateMachineController):
    """ This class logs the usage of different states. This means that it keeps a dictionary which counts the number
        of times the robot is in one state.
    """

    def __init__(self):
        StateMachineController.__init__(self)
        self.state_logger = dict()

    def reset(self, genome, config):
        StateMachineController.reset(self, genome, config)
        self.state_logger = dict()

    def step(self, observation):
        actions = StateMachineController.step(self, observation)

        # Keep a log of the state the robot is in.
        if self.current_state not in self.state_logger:
            self.state_logger[self.current_state] = 0
        self.state_logger[self.current_state] += 1

        return actions
