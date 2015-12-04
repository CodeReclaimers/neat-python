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

