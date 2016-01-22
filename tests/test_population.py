import os
import tempfile

from neat import activation_functions
from neat.config import Config
from neat.population import Population
from neat.statistics import get_average_fitness


def test_minimal():
    # sample fitness function
    def eval_fitness(population):
        for individual in population:
            individual.fitness = 1.0

    # creates the population
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'test_configuration'))

    pop = Population(config)
    # run the simulation for up to 20 generations
    pop.run(eval_fitness, 20)

    # get statistics
    avg_fitness = get_average_fitness(pop)
    assert len(avg_fitness) == 1
    assert all(f == 1 for f in avg_fitness)

    # Change fitness threshold and do another run.
    config.max_fitness_threshold = 1.1
    pop = Population(config)
    # runs the simulation for 20 generations
    pop.run(eval_fitness, 20)

    # get statistics
    avg_fitness = get_average_fitness(pop)
    assert len(avg_fitness) == 20
    assert all(f == 1 for f in avg_fitness)


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
            for act in activation_functions.functions.keys():
                config.allowed_activation = [act]
                for ff in (0, 1):
                    config.feedforward = ff

                    pop = Population(config)
                    pop.run(eval_fitness, 250)


def test_checkpoint():
    # sample fitness function
    def eval_fitness(population):
        for individual in population:
            individual.fitness = 0.99

    # creates the population
    local_dir = os.path.dirname(__file__)
    config = Config(os.path.join(local_dir, 'test_configuration'))

    pop = Population(config)
    pop.run(eval_fitness, 20)

    t = tempfile.NamedTemporaryFile(delete=False)
    t.close()
    pop.save_checkpoint(t.name)

    pop2 = Population(config)
    pop2.load_checkpoint(t.name)

    assert pop.generation_statistics == pop2.generation_statistics
    assert id(pop.generation_statistics) != id(pop2.generation_statistics)
