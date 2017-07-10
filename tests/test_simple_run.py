import os
import neat


def eval_dummy_genome_nn(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return 1.0


def eval_dummy_genomes_nn(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_dummy_genome_nn(genome, config)


def test_serial():
    """tests normal evolution and evaluation"""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, 5))

    # Run for up to 300 generations.
    p.run(eval_dummy_genomes_nn, 300)

    stats.save()
    # stats.save_genome_fitness(with_cross_validation=True)

    stats.get_fitness_stdev()
    # stats.get_average_cross_validation_fitness()
    stats.best_unique_genomes(5)
    stats.best_genomes(5)
    stats.best_genome()

    p.remove_reporter(stats)


def test_parallel():
    """tests parallel evolution and evaluation"""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(4, eval_dummy_genome_nn)
    p.run(pe.evaluate, 300)

    stats.save()


def test_threaded_evaluation():
    """tests a neat evolution using neat.threaded.ThreadedEvaluator"""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    pe = neat.ThreadedEvaluator(4, eval_dummy_genome_nn)
    p.run(pe.evaluate, 300)

    stats.save()


def test_threaded_evaluator():
    """tests generall functionality of neat.threaded.ThreadedEvaluator"""
    n_workers = 3
    e = neat.ThreadedEvaluator(n_workers, eval_dummy_genome_nn)
    # ensure workers are not started
    if (len(e.workers) > 0) or (e.working):
        raise Exception("ThreadedEvaluator started on __init__()!")
    # ensure start() starts the workers
    e.start()
    if (len(e.workers) != n_workers) or (not e.working):
        raise Exception("ThreadedEvaluator did not start on start()!")
    w = e.workers[0]
    if not w.is_alive():
        raise Exception("Workers did not start on start()")
    # ensure a second call to start() does nothing when already started
    e.start()
    if (len(e.workers) != n_workers) or (not e.working):
        raise Exception(
            "A second ThreadedEvaluator.start() call was not ignored!"
            )
    w = e.workers[0]
    if not w.is_alive():
        raise Exception("A worker died or stopped!")
    # ensure stop() works
    e.stop()
    if (len(e.workers) != 0) or (e.working):
        raise Exception(
            "ThreadedEvaluator.stop() did not work!"
            )
    if w.is_alive():
        raise Exception("A worker is still alive!")
    # ensure a second call to stop() does nothing when already stopped
    e.stop()
    if (len(e.workers) != 0) or (e.working):
        raise Exception(
            "A second ThreadedEvaluator.stop() call was not ignored!"
            )
    if w.is_alive():
        raise Exception("A worker is still alive or was resurrected!")
    # ensure a restart is possible
    # ensure start() starts the workers
    e.start()
    if (len(e.workers) != n_workers) or (not e.working):
        raise Exception("ThreadedEvaluator did not restart on start()!")
    w = e.workers[0]
    if not w.is_alive():
        raise Exception("Workers did not start on start()")
    # ensure del stops workers
    w = e.workers[0]
    del e
    if w.is_alive():
        raise Exception("__del__() did not stop workers!")


def eval_dummy_genomes_nn_recurrent(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        genome.fitness = 1.0


def test_run_nn_recurrent():
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    config.feed_forward = False

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, 5))

    # Run for up to 300 generations.
    p.run(eval_dummy_genomes_nn_recurrent, 30)

    stats.save()


def eval_dummy_genomes_ctrnn(genomes, config):
    for genome_id, genome in genomes:
        net = neat.ctrnn.CTRNN.create(genome, config, 0.01)
        genome.fitness = 1.0


def test_run_ctrnn():
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    config.feed_forward = False

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, 5))

    # Run for up to 300 generations.
    p.run(eval_dummy_genomes_ctrnn, 30)

    stats.save()


def eval_dummy_genomes_iznn(genomes, config):
    for genome_id, genome in genomes:
        net = neat.iznn.IZNN.create(genome, config)
        genome.fitness = 1.0


def test_run_iznn():
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration_iznn')
    config = neat.Config(neat.iznn.IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, 5))

    # Run for up to 300 generations.
    p.run(eval_dummy_genomes_iznn, 30)

    stats.save()


if __name__ == '__main__':
    test_serial()
    test_threaded_evaluator()
    test_threaded_evaluation()
    test_parallel()