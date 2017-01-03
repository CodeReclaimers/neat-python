import os
import tempfile

from neat.indexer import Indexer


# def test_minimal():
#     # sample fitness function
#     def eval_fitness(population):
#         for individual in population:
#             individual.fitness = 1.0
#
#     # creates the population
#     local_dir = os.path.dirname(__file__)
#     config = Config(os.path.join(local_dir, 'test_configuration'))
#     config.save_best = True
#
#     pop = Population(config)
#     # run the simulation for up to 20 generations
#     pop.run(eval_fitness, 20)
#
#     # Test save_checkpoint with defaults
#     pop.save_checkpoint()
#
#     # get statistics
#     best_unique = pop.statistics.best_unique_genomes(1)
#     best_unique = pop.statistics.best_unique_genomes(2)
#     avg_fitness = pop.statistics.get_average_fitness()
#     assert len(avg_fitness) == 1
#     assert all(f == 1 for f in avg_fitness)
#
#     # Change fitness threshold and do another run.
#     config.max_fitness_threshold = 1.1
#     pop = Population(config)
#     # runs the simulation for 20 generations
#     pop.run(eval_fitness, 20)
#
#     # get statistics
#     avg_fitness = pop.statistics.get_average_fitness()
#     assert len(avg_fitness) == 20
#     assert all(f == 1 for f in avg_fitness)
#
#
# def test_config_options():
#     # sample fitness function
#     def eval_fitness(population):
#         for individual in population:
#             individual.fitness = 1.0
#
#     local_dir = os.path.dirname(__file__)
#     config = Config(os.path.join(local_dir, 'test_configuration'))
#
#     for hn in (0, 1, 2):
#         config.hidden_nodes = hn
#         for fc in (0, 1):
#             #config.fully_connected = fc
#             for act in activation_functions.functions.keys():
#                 config.allowed_activation = [act]
#                 for ff in (0, 1):
#                     config.feedforward = ff
#
#                     pop = Population(config)
#                     pop.run(eval_fitness, 250)


# def test_checkpoint():
#     # sample fitness function
#     def eval_fitness(population):
#         for individual in population:
#             individual.fitness = 0.99
#
#     # creates the population
#     local_dir = os.path.dirname(__file__)
#     config = Config(os.path.join(local_dir, 'test_configuration'))
#
#     pop = Population(config)
#     pop.run(eval_fitness, 20)
#
#     t = tempfile.NamedTemporaryFile(delete=False)
#     t.close()
#     pop.save_checkpoint(t.name)
#
#     pop2 = Population(config)
#     pop2.load_checkpoint(t.name)
#
#     #assert pop.species == pop2.species
#     #assert id(pop.species) != id(pop2.species)
#
#     # assert pop.statistics.generation_statistics == pop2.statistics.generation_statistics
#     # assert id(pop.statistics.generation_statistics) != id(pop2.statistics.generation_statistics)


def test_indexer():
    indexer0 = Indexer(0)
    assert indexer0.get_next() == 0
    assert indexer0.get_next() == 1
    assert indexer0.get_next() == 2


    indexer17 = Indexer(17)
    assert indexer17.get_next() == 17
    assert indexer17.get_next() == 18
    assert indexer17.get_next() == 19


if __name__ == '__main__':
    test_indexer()