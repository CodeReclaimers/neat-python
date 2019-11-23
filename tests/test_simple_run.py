from __future__ import print_function

import multiprocessing
import os

import neat

VERBOSE = True


def eval_dummy_genome_nn(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ignored_output = net.activate((0.5, 0.5))
    return 0.0


def eval_dummy_genomes_nn(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_dummy_genome_nn(genome, config)


def test_serial():
    """Test basic (dummy fitness function) non-parallel run."""
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

    # Run for up to 19 generations.
    p.run(eval_dummy_genomes_nn, 19)

    stats.save()
    # stats.save_genome_fitness(with_cross_validation=True)

    assert len(stats.get_fitness_stdev())
    stats.best_unique_genomes(5)
    stats.best_genomes(5)
    stats.best_genome()

    p.remove_reporter(stats)


def eval_dummy_genome_nn_bad(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ignored_output = net.activate((0.5, 0.5, 0.5))
    return 0.0


def eval_dummy_genomes_nn_bad(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_dummy_genome_nn_bad(genome, config)


def test_serial_bad_input():
    """Make sure get error for bad input."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    try:
        p.run(eval_dummy_genomes_nn_bad, 45)
    except Exception:  # may change in nn.feed_forward code to more specific...
        pass
    else:
        raise Exception("Did not get Exception from bad input")


def test_serial_random():
    """Test basic (dummy fitness function) non-parallel run w/random activation, aggregation init."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration2')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    if VERBOSE:
        print("config.genome_config.__dict__: {!r}".format(
            config.genome_config.__dict__))

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(VERBOSE))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(15, 1))

    # Run for up to 45 generations.
    p.run(eval_dummy_genomes_nn, 45)

    stats.save()
    # stats.save_genome_fitness(with_cross_validation=True)

    stats.get_fitness_stdev()
    stats.best_unique_genomes(5)
    stats.best_genomes(5)
    stats.best_genome()

    p.remove_reporter(stats)


def test_serial3():
    """Test more configuration variations for simple serial run."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration3')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    if VERBOSE:
        print("config.genome_config.__dict__: {!r}".format(
            config.genome_config.__dict__))

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(VERBOSE))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(15, 1))

    # Run for up to 45 generations.
    p.run(eval_dummy_genomes_nn, 45)

    stats.save()
    # stats.save_genome_fitness(with_cross_validation=True)

    stats.get_fitness_stdev()
    stats.best_unique_genomes(5)
    stats.best_genomes(5)
    stats.best_genome()

    p.remove_reporter(stats)


def test_serial4():
    """Test more configuration variations for simple serial run."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration4')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    if VERBOSE:
        print("config.genome_config.__dict__: {!r}".format(
            config.genome_config.__dict__))

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(VERBOSE))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(15, 1))

    # Run for up to 45 generations.
    p.run(eval_dummy_genomes_nn, 45)

    stats.save()
    # stats.save_genome_fitness(with_cross_validation=True)

    stats.get_fitness_stdev()
    stats.best_unique_genomes(5)
    stats.best_genomes(5)
    stats.best_genome()

    p.remove_reporter(stats)


def test_serial5():
    """Test more configuration variations for simple serial run."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration5')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    if VERBOSE:
        print("config.genome_config.__dict__: {!r}".format(
            config.genome_config.__dict__))

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(VERBOSE))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(15, 1))

    # Run for up to 45 generations.
    p.run(eval_dummy_genomes_nn, 45)

    stats.save()
    # stats.save_genome_fitness(with_cross_validation=True)

    stats.get_fitness_stdev()
    stats.best_unique_genomes(5)
    stats.best_genomes(5)
    stats.best_genome()

    p.remove_reporter(stats)


def test_serial4_bad():
    """Make sure no_fitness_termination and n=None give an error."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration4')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    if VERBOSE:
        print("config.genome_config.__dict__: {!r}".format(
            config.genome_config.__dict__))

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    try:
        p.run(eval_dummy_genomes_nn, None)
    except RuntimeError:
        pass
    else:
        raise Exception(
            "Should have had a RuntimeError with n=None and no_fitness_termination")


def test_serial_bad_config():
    """Test if bad_configuration1 causes a LookupError or TypeError on trying to run."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'bad_configuration1')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    try:
        p.run(eval_dummy_genomes_nn, 19)
    except (LookupError, TypeError):
        pass
    else:
        raise Exception(
            "Should have had a LookupError/TypeError with bad_configuration1")


def test_serial_bad_configA():
    """Test if bad_configurationA causes a RuntimeError on trying to create the population."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'bad_configurationA')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    try:
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)
    except RuntimeError:
        pass
    else:
        raise Exception(
            "Should have had a RuntimeError with bad_configurationA")


def test_serial_extinction_exception():
    """Test for complete extinction with exception."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    config.stagnation_config.max_stagnation = 1
    config.stagnation_config.species_elitism = 0

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))

    try:
        # Run for up to 45 generations.
        p.run(eval_dummy_genomes_nn, 45)
    except Exception:
        pass
    else:
        raise Exception("Should have had a complete extinction at some point!")


def test_serial_extinction_no_exception():
    """Test for complete extinction without exception."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    config.stagnation_config.max_stagnation = 1
    config.stagnation_config.species_elitism = 0
    config.reset_on_extinction = True

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    reporter = neat.StdOutReporter(True)
    p.add_reporter(reporter)
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 45 generations.
    p.run(eval_dummy_genomes_nn, 45)

    assert reporter.num_extinctions > 0, "No extinctions happened!"

    stats.save()
    p.remove_reporter(stats)


def test_parallel():
    """Test parallel run using ParallelEvaluator (subprocesses)."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(VERBOSE))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, 5))

    # Run for up to 19 generations.
    pe = neat.ParallelEvaluator(1 + multiprocessing.cpu_count(), eval_dummy_genome_nn)
    p.run(pe.evaluate, 19)

    stats.save()


def test_threaded_evaluation():
    """Tests a neat evolution using neat.threaded.ThreadedEvaluator"""
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

    # Run for up to 19 generations.
    pe = neat.ThreadedEvaluator(4, eval_dummy_genome_nn)
    p.run(pe.evaluate, 19)

    stats.save()


def test_threaded_evaluator():
    """Tests general functionality of neat.threaded.ThreadedEvaluator"""
    n_workers = 3
    e = neat.ThreadedEvaluator(n_workers, eval_dummy_genome_nn)
    try:
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
    finally:  # try to close if KeyboardInterrupt or similar
        if len(e.workers) or e.working:
            e.stop()
    # ensure del stops workers
    del e
    # unfortunately, __del__() may never be called, even when using del
    # this means that testing for __del__() to call stop() may not work
    # this test had a high random failure rate, so i removed it.
    # if w.is_alive():
    #     raise Exception("__del__() did not stop workers!")


def eval_dummy_genomes_nn_recurrent(genomes, config):
    for ignored_genome_id, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        ignored_output = net.activate((0.5, 0.5))
        net.reset()
        genome.fitness = 0.0


def test_run_nn_recurrent():
    """Basic test of nn.recurrent function."""
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
    p.add_reporter(neat.StdOutReporter(VERBOSE))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, 5))

    # Run for up to 19 generations.
    p.run(eval_dummy_genomes_nn_recurrent, 19)

    stats.save()


def eval_dummy_genomes_nn_recurrent_bad(genomes, config):
    for ignored_genome_id, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        ignored_output = net.activate((0.5, 0.5, 0.5))
        net.reset()
        genome.fitness = 0.0


def test_run_nn_recurrent_bad():
    """Make sure nn.recurrent gives error on bad input."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    config.feed_forward = False

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    try:
        p.run(eval_dummy_genomes_nn_recurrent_bad, 19)
    except Exception:  # again, may change to more specific in nn.recurrent
        pass
    else:
        raise Exception("Did not get Exception for bad input to nn.recurrent")


def eval_dummy_genomes_ctrnn(genomes, config):
    for genome_id, genome in genomes:
        net = neat.ctrnn.CTRNN.create(genome, config, 0.01)
        if genome_id <= 150:
            genome.fitness = 0.0
        else:
            net.reset()
            genome.fitness = 1.0


def test_run_ctrnn():
    """Basic test of continuous-time recurrent neural network (ctrnn)."""
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
    p.add_reporter(neat.StdOutReporter(VERBOSE))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, 5))

    # Run for up to 19 generations.
    p.run(eval_dummy_genomes_ctrnn, 19)

    stats.save()

    unique_genomes = stats.best_unique_genomes(5)
    assert 1 <= len(unique_genomes) <= 5, "Unique genomes: {!r}".format(unique_genomes)
    genomes = stats.best_genomes(5)
    assert 1 <= len(genomes) <= 5, "Genomes: {!r}".format(genomes)
    stats.best_genome()

    p.remove_reporter(stats)


def eval_dummy_genomes_ctrnn_bad(genomes, config):
    for genome_id, genome in genomes:
        net = neat.ctrnn.CTRNN.create(genome, config, 0.01)
        net.advance([0.5, 0.5, 0.5], 0.01, 0.05)
        if genome_id <= 150:
            genome.fitness = 0.0
        else:
            net.reset()
            genome.fitness = 1.0


def test_run_ctrnn_bad():
    """Make sure ctrnn gives error on bad input."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    config.feed_forward = False

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    try:
        p.run(eval_dummy_genomes_ctrnn_bad, 19)
    except RuntimeError:
        pass
    else:
        raise Exception("Did not get RuntimeError for bad input to ctrnn")


def eval_dummy_genomes_iznn(genomes, config):
    for genome_id, genome in genomes:
        net = neat.iznn.IZNN.create(genome, config)
        if genome_id < 10:
            net.reset()
            genome.fitness = 0.0
        elif genome_id <= 150:
            genome.fitness = 0.5
        else:
            genome.fitness = 1.0


def test_run_iznn():
    """
    Basic test of spiking neural network (iznn).
    [TODO: Takes the longest of any of the tests in this file, by far. Why?]
    Was because had population of 290 thanks to too much speciation -
    too-high compatibility_weight_coefficient relative to range for weights.
    """
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
    p.add_reporter(neat.Checkpointer(2, 10))

    # Run for up to 20 generations.
    p.run(eval_dummy_genomes_iznn, 20)

    stats.save()

    unique_genomes = stats.best_unique_genomes(5)
    assert 1 <= len(unique_genomes) <= 5, "Unique genomes: {!r}".format(unique_genomes)
    genomes = stats.best_genomes(5)
    assert len(genomes) == 5, "Genomes: {!r}".format(genomes)
    stats.best_genome()

    p.remove_reporter(stats)


def eval_dummy_genomes_iznn_bad(genomes, config):
    for genome_id, genome in genomes:
        net = neat.iznn.IZNN.create(genome, config)
        net.set_inputs([0.5, 0.5, 0.5])
        if genome_id < 10:
            net.reset()
            genome.fitness = 0.0
        elif genome_id <= 150:
            genome.fitness = 0.5
        else:
            genome.fitness = 1.0


def test_run_iznn_bad():
    """Make sure iznn gives error on bad input."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration_iznn')
    config = neat.Config(neat.iznn.IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    try:
        p.run(eval_dummy_genomes_iznn_bad, 19)
    except RuntimeError:
        pass
    else:
        raise Exception("Did not get RuntimeError for bad input to iznn")


if __name__ == '__main__':
    VERBOSE = False
    test_serial()
    test_serial_random()
    test_serial3()
    test_serial4()
    test_serial5()
    test_serial_bad_config()
    test_serial_bad_configA()
    test_serial_extinction_exception()
    test_serial_extinction_no_exception()
    test_parallel()
    test_threaded_evaluation()
    test_threaded_evaluator()
    test_run_nn_recurrent()
    test_run_nn_recurrent_bad()
    test_run_ctrnn()
    test_run_ctrnn_bad()
    test_run_iznn()
    test_run_iznn_bad()
