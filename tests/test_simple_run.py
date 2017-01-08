from __future__ import print_function

import os
import neat


def eval_dummy_genome(genome, config):
    return 1.0


def eval_dummy_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_dummy_genome(genome, config)


def test_serial():
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter())
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    p.run(eval_dummy_genomes, 300)

    stats.save()


def test_parallel():
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter())
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(4, eval_dummy_genome)
    p.run(pe.evaluate, 300)

    stats.save()

# def test_run():
#     xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
#     xor_outputs = [0, 1, 1, 0]
#
#     def eval_fitness(genomes):
#         for g in genomes:
#             net = nn.create_feed_forward_phenotype(g)
#
#             error = 0.0
#             for inputs, expected in zip(xor_inputs, xor_outputs):
#                 # Serial activation propagates the inputs through the entire network.
#                 output = net.serial_activate(inputs)
#                 error += (output[0] - expected) ** 2
#
#             # When the output matches expected for all inputs, fitness will reach
#             # its maximum value of 1.0.
#             g.fitness = 1 - error
#
#     local_dir = os.path.dirname(__file__)
#     config = Config(os.path.join(local_dir, 'test_configuration'))
#
#     pop = population.Population(config)
#     pop.run(eval_fitness, 10)
#
#     winner = pop.statistics.best_genome()
#
#     # Validate winner.
#     for g in pop.statistics.most_fit_genomes:
#         assert winner.fitness >= g.fitness
#
#     statistics.save_stats(pop.statistics)
#     statistics.save_species_count(pop.statistics)
#     statistics.save_species_fitness(pop.statistics)
#

if __name__ == '__main__':
    test_serial()
    test_parallel()