import neat
from neat.state_machine_network import StateMachineNetwork


class SwarmExperimentRunner:
    """ This class represents an experiment runner, so an instance which can be used to run gym experiments.
        Note that the fitness is only requested using the get_fitness() function of the environment after the last step.
        So make sure the gym environment provides this.
    """

    def __init__(self, gym_environment, num_steps):
        self.env = gym_environment
        self.num_steps = num_steps

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


class NEATSwarmExperimentRunner(SwarmExperimentRunner):
    """ This class can be used to run Neat experiments."""

    def run(self, genome, config):
        """ This function runs an experiment for a NEAT genome, given a genome and the required variables."""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation = self.env.reset()

        for i in range(self.num_steps):
            output = [net.activate(observation[i]) for i in range(len(observation))]
            observation, _, _, _ = self.env.step(output)

        return self.env.get_fitness()


class SMSwarmExperimentRunner(SwarmExperimentRunner):
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
            observation, _, _, _ = self.env.step(actions)

        return self.env.get_fitness()
