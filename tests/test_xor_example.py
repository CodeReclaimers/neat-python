from __future__ import print_function

import os

import neat


def test_xor_example_uniform_weights():
    test_xor_example(uniform_weights=True)


def test_xor_example(uniform_weights=False):
    # 2-input XOR inputs and expected outputs.
    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 1.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for xi, xo in zip(xor_inputs, xor_outputs):
                output = net.activate(xi)
                genome.fitness -= (output[0] - xo[0]) ** 2

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    if uniform_weights:
        config.genome_config.weight_init_type = 'uniform'
        filename_prefix = 'neat-checkpoint-test_xor_uniform-'
    else:
        filename_prefix = 'neat-checkpoint-test_xor-'

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    checkpointer = neat.Checkpointer(25, 10, filename_prefix)
    p.add_reporter(checkpointer)

    # Run for up to 100 generations, allowing extinction.
    winner = None
    try:
        winner = p.run(eval_genomes, 100)
    except neat.CompleteExtinctionException as e:
        pass

    assert len(stats.get_fitness_median()), "Nothing returned from get_fitness_median()"

    if winner:
        if uniform_weights:
            print('\nUsing uniform weight initialization:')
        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = winner_net.activate(xi)
            print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    if (checkpointer.last_generation_checkpoint >= 0) and (checkpointer.last_generation_checkpoint < 100):
        filename = '{0}{1}'.format(filename_prefix, checkpointer.last_generation_checkpoint)
        print("Restoring from {!s}".format(filename))
        p2 = neat.checkpoint.Checkpointer.restore_checkpoint(filename)
        p2.add_reporter(neat.StdOutReporter(True))
        stats2 = neat.StatisticsReporter()
        p2.add_reporter(stats2)

        winner2 = None
        try:
            winner2 = p2.run(eval_genomes, (100 - checkpointer.last_generation_checkpoint))
        except neat.CompleteExtinctionException:
            pass

        if winner2:
            if not winner:
                raise Exception("Had winner2 without first-try winner")
        elif winner:
            raise Exception("Had first-try winner without winner2")


if __name__ == '__main__':
    test_xor_example()
    test_xor_example_uniform_weights()
