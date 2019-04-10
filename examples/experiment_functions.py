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
        controller = self.controller_class()
        controller.reset(genome, config)
        observation = self.env.reset()
        fitness = 0

        for i in range(self.num_steps):

            output = controller.step(observation)
            observation, fitness, done, _ = self.env.step(output)

            self.check_render()
            if done:
                break

        return fitness

    def draw(self, genome, config):
        """ This function should draw the given genome. It depends on the genome that is actually used."""
        controller = self.controller_class()
        controller.reset(genome, config)
        controller.draw()

    def check_render(self):
        if self.render:
            self.env.render()


class SwarmExperimentRunner(ExperimentRunner):
    """ This class describes an experiment runner for a swarm, ie. multiple robots with the same controller."""

    def run(self, genome, config):

        observations = self.env.reset()

        # Spawn as many controllers as there are observations.
        controllers = [self.controller_class() for _ in range(len(observations))]

        # Reset all controllers with the genome and config.
        for controller in controllers:
            controller.reset(genome, config)

        for i in range(self.num_steps):

            output = [controllers[i].step(observations[i]) for i in range(len(observations))]
            observations, _, done, _ = self.env.step(output)

            self.check_render(i)
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

    def draw(self):
        """ This function draws the current controller that is being used."""
        pass


class FeedForwardNetworkController(SimulationController):
    """ This class calculates the next actions based on a feed forward network."""

    def reset(self, genome, config):
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def step(self, observation):
        return self.net.activate(observation)


class StateMachineController(SimulationController):
    """ This class calculates the next actions based on a state machine. The difference here is that the current state,
        needs to be taken into account and updated, as this is also part of the state machine.
    """

    def __init__(self):
        super().__init__()
        self.current_state = 0

    def reset(self, genome, config):
        self.net = StateMachineNetwork.create(genome, config.genome_config)
        self.current_state = 0

    def step(self, observation):
        new_state, actions = self.net.activate(self.current_state, observation)
        self.current_state = new_state

        return actions
