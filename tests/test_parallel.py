import os
from neat.population import Population
from neat import parallel


# dummy fitness function
def eval_fitness(individual):
    return 1.0


def test_minimal():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')

    pop = Population(config_path)
    pe = parallel.ParallelEvaluator(4, eval_fitness)
    pop.run(pe.evaluate, 400)
