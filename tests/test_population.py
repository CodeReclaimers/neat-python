from neat.population import Population
from neat.config import Config


def test_minimal():
    # sample fitness function
    def eval_fitness(population):
        for individual in population:
            individual.fitness = 1.0

    # creates the population
    config = Config('test_configuration')

    pop = Population(config)
    # runs the simulation for 250 epochs
    pop.epoch(eval_fitness, 250)

