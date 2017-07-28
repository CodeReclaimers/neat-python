from __future__ import print_function

import os

import neat

def test_xor_example_multiparam_relu():
    test_xor_example(activation_default='multiparam_relu')

def test_xor_example_multiparam_sigmoid_or_relu():
    test_xor_example(uniform_weights=True,
                     activation_default='random',
                     activation_options=['multiparam_sigmoid','relu'])

def test_xor_example_multiparam_aggregation():
    test_xor_example(uniform_weights=True,
                     activation_default='multiparam_sigmoid',
                     aggregation_default='random',
                     aggregation_options=['sum','max_median_min','maxabs_mean'])

def test_xor_example_uniform_weights():
    test_xor_example(uniform_weights=True)

def test_xor_example(uniform_weights=False, activation_default=None, activation_options=None,
                     aggregation_default=None, aggregation_options=None):
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

    if activation_default is not None:
        config.genome_config.activation_default = activation_default
        if activation_options is None:
            config.genome_config.activation_options = [activation_default]

    if activation_options is not None:
        config.genome_config.activation_options = activation_options
        if len(activation_options) > 1:
            config.genome_config.activation_mutate_rate = 0.1

    if aggregation_default is not None:
        config.genome_config.aggregation_default = aggregation_default
        if aggregation_options is None:
            config.genome_config.aggregation_options = [aggregation_default]

    if aggregation_options is not None:
        config.genome_config.aggregation_options = aggregation_options
        if len(aggregation_options) > 1:
            config.genome_config.aggregation_mutate_rate = 0.1

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    checkpointer = neat.Checkpointer(25, 10)
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
        filename = 'neat-checkpoint-{0}'.format(checkpointer.last_generation_checkpoint)
        print("Restoring from {!s}".format(filename))
        p2 = neat.checkpoint.Checkpointer.restore_checkpoint(filename)
        p2.add_reporter(neat.StdOutReporter(True))
        stats2 = neat.StatisticsReporter()
        p2.add_reporter(stats2)

        winner2 = None
        try:
            winner2 = p2.run(eval_genomes, (100-checkpointer.last_generation_checkpoint))
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
    test_xor_example_multiparam_relu()
    test_xor_example_multiparam_sigmoid_or_relu()
    test_xor_example_multiparam_aggregation()
