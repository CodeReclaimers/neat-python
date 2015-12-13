import os
from neat.population import Population
from neat.config import Config


def test_minimal():
    # sample fitness function
    def eval_fitness(population):
        for individual in population:
            individual.fitness = 1.0

    # creates the population
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'test_configuration'))

    pop = Population(config)
    # runs the simulation for 250 epochs
    pop.epoch(eval_fitness, 250)


def test_config_options():
    # sample fitness function
    def eval_fitness(population):
        for individual in population:
            individual.fitness = 1.0

    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'test_configuration'))

    for hn in (0, 1, 2):
        config.hidden_nodes = hn
        for fc in (0, 1):
            config.fully_connected = fc
            for act in ('exp', 'tanh'):
                config.nn_activation = act
                for ff in (0, 1):
                    config.feedforward = ff

                    pop = Population(config)
                    pop.epoch(eval_fitness, 250)
