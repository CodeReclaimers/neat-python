"""
A distributed version of XOR using neat.distributed.

Since XOR is a simple experiment, a distributed version probably won't run any
faster than the single-process version, due to the overhead of
inter-process communication.

If your evaluation function is what's taking up most of your processing time
(and you should check by using a profiler while running single-process),
you should see a significant performance improvement by evaluating across
multiple machines.

This example is only intended to show how to do a distributed experiment
in neat-python.  You can of course roll your own parallelism mechanism
or inherit from DistributedEvaluator if you need to do something more
complicated.
"""

from __future__ import print_function

import os
import argparse
import sys

import neat

import visualize

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,),     (1.0,),     (1.0,),     (0.0,)]


def eval_genome(genome, config):
    """
    This function will be run in parallel by ParallelEvaluator.  It takes two
    arguments (a single genome and the genome class configuration data) and
    should return one float (that genome's fitness).

    Note that this function needs to be in module scope for multiprocessing.Pool
    (which is what DistributedEvaluator uses when using more than one worker)
    to find it.  Because of this, make sure you check for __main__ before
    executing any code (as we do here in the last few lines in the file),
    otherwise you'll have made a fork bomb instead of a neuroevolution demo. :)
    """

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    error = 4.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        error -= (output[0] - xo[0]) ** 2
    return error


def run(config_file, addr, authkey, mode, workers):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # setup an DistributedEvaluator
    de = neat.DistributedEvaluator(
        addr,  # connect to addr
        authkey,  # use authkey to authenticate
        eval_genome,  # use eval_genome() to evaluate a genome
        secondary_chunksize=4,  # send 4 genomes at once
        num_workers=workers,  # when in secondary mode, use this many workers
        worker_timeout=10,  # when in secondary mode and workers > 1,
                            # wait at most 10 seconds for the result
        mode=mode,  # whether this is the primary or a secondary node
                    # in most case you can simply pass
                    # 'neat.distributed.MODE_AUTO' as the mode.
                    # This causes the DistributedEvaluator to
                    # determine the mode by checking if address
                    # points to the localhost.
        )

    # start the DistributedEvaluator
    de.start(
        exit_on_stop=True,  # if this is a secondary node, call sys.exit(0) when
                            # when finished. All code after this line will only
                            # be executed by the primary node.
        secondary_wait=3,  # when a secondary, sleep this many seconds before continuing
                       # this is useful when the primary node may need more time
                       # to start than the secondary nodes.
        )

    # Run for up to 500 generations.
    winner = p.run(de.evaluate, 500)

    # stop evaluator
    de.stop()

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


def addr_tuple(s):
    """converts a string into a tuple of (host, port)"""
    if "[" in s:
        # ip v6
        if (s.count("[") != 1) or (s.count("]") != 1):
            raise ValueError("Invalid IPv6 address!")
        end = s.index("]")
        if ":" not in s[end:]:
            raise ValueError("IPv6 address does specify a port to use!")
        host, port = s[1:end], s[end+1:]
        port = int(port)
        return (host, port)
    else:
        host, port = s.split(":")
        port = int(port)
        return (host, port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NEAT xor experiment evaluated across multiple machines.")
    parser.add_argument(
        "address",
        help="host:port address of the main node",
        type=addr_tuple,
        action="store",
        )
    parser.add_argument(
        "--workers",
        type=int,
        help="number of processes to use for evaluating the genomes",
        action="store",
        default=1,
        dest="workers",
        )
    parser.add_argument(
        "--authkey",
        action="store",
        help="authkey to use (default: 'neat-python')",
        default="neat-python",
        dest="authkey",
        )
    parser.add_argument(
        "--force-secondary","--force-slave",
        action="store_const",
        const=neat.distributed.MODE_SECONDARY,
        default=neat.distributed.MODE_AUTO,
        help="Force secondary mode (useful for debugging)",
        dest="mode",
        )
    ns = parser.parse_args()

    address = ns.address
    host, port = address
    workers = ns.workers
    authkey = ns.authkey
    mode = ns.mode

    if (host in ("0.0.0.0", "localhost", "")) and (mode == neat.distributed.MODE_AUTO):
        # print an error message
        # we are using auto-mode determination in this example,
        # which does not work well with '0.0.0.0' or 'localhost'.
        print("Please do not use '0.0.0.0' or 'localhost' as host.")
        sys.exit(1)

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')

    print("Starting Node...")
    print("Please ensure that you are using more than one node.")

    run(config_path, address, authkey, mode, workers)
