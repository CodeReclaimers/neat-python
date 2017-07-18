"""tests for neat.distributed"""
from __future__ import print_function

import socket
import os
import multiprocessing
import random
import sys
import threading

import neat
from neat.distributed import chunked, MODE_AUTO, MODE_MASTER, MODE_SLAVE, RoleError


def eval_dummy_genome_nn(genome, config):
    """dummy evaluation function"""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return 1.0


def test_chunked():
    """test for neat.distributed.chunked"""
    # test chunked(range(110), 10)
    # => 11 chunks of 10 elements
    d110 = list(range(110))
    d110c = chunked(d110, 10)
    if len(d110c) != 11:
        raise Exception("neat.distributed.chunked(range(110), 10) did not create 11 chunks!")
    if len(d110c[0]) != 10:
        raise Exception("neat.distributed.chunked(range(110), 10) did not create chunks of length 10!")
    rec = []
    for e in d110c:
        rec += e
    if rec != d110:
        raise Exception("neat.distributed.chunked(range(110), 10) did create wrong chunks!")
    # test invalid argument checking
    try:
        chunked(range(10), 0)
    except ValueError:
        pass
    else:
        raise Exception("neat.distributed.chunked(range(10), 0) did not raise an exception!")
    try:
        chunked(range(10), 1.1)
    except ValueError:
        pass
    else:
        raise Exception("neat.distributed.chunked(range(10), 1.1) did not raise an exception!")
    # test chunked(range(13, 3))
    # => 4 chunks of 3 elements, 1 chunk of 1 element
    d13 = list(range(13))
    d13c = chunked(d13, 3)
    if len(d13c) != 5:
        raise Exception("neat.distributed.chunked(range(13), 3) did not create 5 chunks!")
    if len(d13c[0]) != 3:
        raise Exception("neat.distributed.chunked(range(13), 3) did not create chunks of length 3!")
    if len(d13c[-1]) != 1:
        raise Exception("neat.distributed.chunked(range(13), 3) did not create  a last chunk of length 1!")
    rec = []
    for e in d13c:
        rec += e
    if rec != d13:
        raise Exception("neat.distributed.chunked(range(13), 3) did create wrong chunks!")


def test_host_is_local():
    """test for neat.distributed.host_is_local"""
    tests = (
        # (hostname or ip, expected value)
        ("localhost", True),
        ("0.0.0.0", True),
        ("127.0.0.1", True),
        #("::1", True), # depends on IP, etc setup on host to work right
        (socket.gethostname(), True),
        (socket.getfqdn(), True),
        ("github.com", False),
        ("google.de", False),
        )
    for hostname, islocal in tests:
        try:
            result = neat.host_is_local(hostname)
        except EnvironmentError:
            print("test_host_is_local: Error with hostname {0!r} (expected {1!r})".format(hostname,
                                                                                          islocal))
            raise
        else: # if do not want to do 'raise' above for some cases
            if result != islocal:
                raise Exception(
                    "Hostname/IP: {h}; Expected: {e}; Got: {r}".format(
                        h=hostname, e=islocal, r=result,
                        )
                    )


def test_DistributedEvaluator_mode():
    """tests for the mode determination of DistributedEvaluator"""
    # test auto mode setting
    # we also test that the mode is not
    # automatically determined when explicitly
    # given
    tests = (
        # (hostname or ip, mode to pass, expected mode)
        ("localhost", MODE_MASTER, MODE_MASTER),
        ("0.0.0.0", MODE_MASTER, MODE_MASTER),
        ("localhost", MODE_SLAVE, MODE_SLAVE),
        ("example.org", MODE_MASTER, MODE_MASTER),
        (socket.gethostname(), MODE_SLAVE, MODE_SLAVE),
        ("localhost", MODE_AUTO, MODE_MASTER),
        (socket.gethostname(), MODE_AUTO, MODE_MASTER),
        (socket.getfqdn(), MODE_AUTO, MODE_MASTER),
        ("example.org", MODE_AUTO, MODE_SLAVE),
        )
    for hostname, mode, expected in tests:
        addr = (hostname, 80)
        try:
            de = neat.DistributedEvaluator(
                addr,
                authkey=b"abcd1234",
                eval_function=eval_dummy_genome_nn,
                mode=mode,
                )
        except EnvironmentError:
            print("test_DistributedEvaluator_mode(): Error with hostname " +
                  "{!r}".format(hostname))
            raise
        result = de.mode
        if result != expected:
            raise Exception(
                "Mode determination failed! Hostname: {h}; expected: {e}; got: {r}!".format(
                    h=hostname, e=expected, r=result,
                    )
                )
        if result == MODE_AUTO:
            raise Exception(
                "DistributedEvaluator.__init__(mode=MODE_AUTO) did not automatically determine its mode!"
                )
        elif result == MODE_MASTER:
            if not de.is_master():
                raise Exception(
                    "DistributedEvaluator.is_master() returns False even if the evaluator is in master mode!"
                    )
        elif result == MODE_SLAVE:
            if de.is_master():
                raise Exception(
                    "DistributedEvaluator.is_master() returns True even if the evaluator is in slave mode!"
                    )
    # test invalid mode error
    de = neat.DistributedEvaluator(
        addr,
        authkey=b"abcd1234",
        eval_function=eval_dummy_genome_nn,
        mode="#invalid MODE!",
        )
    try:
        de.start()
    except ValueError:
        pass
    else:
        raise Exception("Passing an invalid mode did not cause an exception to be raised on start()!")

def test_DistributedEvaluator_master_restrictions():
    """tests that some master-exclusive methods fail when called  by the slaves"""
    slave = neat.DistributedEvaluator(
        ("localhost", 80),
        authkey=b"abcd1234",
        eval_function=eval_dummy_genome_nn,
        mode=MODE_SLAVE,
        )
    try:
        slave.stop()
    except RoleError:
        # only ignore RoleErrors
        # a RuntimeError should only be raised when in master mode.
        pass
    else:
        raise Exception("A DistributedEvaluator in slave mode could call stop()!")
    try:
        slave.evaluate(None, None)  # we do not need valid values for this test
    except RoleError:
        # only ignore RoleErrors
        # other errors should only be raised when in master mode.
        pass
    else:
        raise Exception("A DistributedEvaluator in slave mode could call evaluate()!")


def test_distributed_evaluation_multiprocessing():
    """
    Full test run using the Distributed Evaluator.
    Note that this is not a very good test for the
    DistributedEvaluator, because we still work on
    one machine, not accross multiple machines.
    We emulate the other machines using subprocesses
    created using the multiprocessing module.
    """
    addr = ("localhost", random.randint(12000, 30000))
    authkey = b"abcd1234"
    mp = multiprocessing.Process(
        name="Master evaluation process",
        target=run_master,
        args=(addr, authkey, 300),
        )
    mp.start()
    mwcp = multiprocessing.Process(
        name="Child evaluation process (multiple workers)",
        target=run_slave,
        args=(addr, authkey, 2),
        )
    swcp = multiprocessing.Process(
        name="Child evaluation process (direct evaluation)",
        target=run_slave,
        args=(addr, authkey, 1),
        )
    swcp.daemon = True  # we cant set this on mwcp
    mwcp.start()
    swcp.start()
    mp.join()
    mwcp.join()
    swcp.join()
    if mp.exitcode != 0:
        raise Exception("Master-process exited with status {s}!".format(s=mp.exitcode))
    if mwcp.exitcode != 0:
        raise Exception("Multiworker-slave-process exited with status {s}!".format(s=mwcp.exitcode))
    if swcp.exitcode != 0:
        raise Exception("Singleworker-slave-process exited with status {s}!".format(s=swcp.exitcode))


def test_distributed_evaluation_threaded():
    """
    Full test run using the Distributed Evaluator.
    Note that this is not a very good test for the
    DistributedEvaluator, because we still work on
    one machine, not accross multiple machines.
    We emulate the other machines using threads.
    This test is like test_distributed_evaluation_multiprocessing,
    but uses threads instead of processes.
    We use this to get the coverage correctly.
    """
    addr = ("localhost", random.randint(12000, 30000))
    authkey = b"abcd1234"
    mp = threading.Thread(
        name="Master evaluation thread",
        target=run_master,
        args=(addr, authkey, 30),
        )
    mp.start()
    mwcp = threading.Thread(
        name="Child evaluation thread (multiple workers)",
        target=run_slave,
        args=(addr, authkey, 2),
        )
    swcp = threading.Thread(
        name="Child evaluation thread (direct evaluation)",
        target=run_slave,
        args=(addr, authkey, 1),
        )
    swcp.daemon = True  # we cant set this on mwcp
    mwcp.start()
    swcp.start()
    mp.join()
    mwcp.join()
    swcp.join()
    
    # we cant check for exceptions in the threads.
    # however, these checks are also done in
    # test_distributed_evaluationmultiprocessing,
    # so they should not fail here.
    # also, this test is mainly for the coverage.

    

def run_master(addr, authkey, generations):
    """starts a DistributedEvaluator in master mode."""
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
    de = neat.DistributedEvaluator(
        addr,
        authkey=authkey,
        eval_function=eval_dummy_genome_nn,
        mode=MODE_MASTER,
        )
    de.start()
    p.run(de.evaluate, generations)
    de.stop(wait=3)

    stats.save()


def run_slave(addr, authkey, num_workers=1):
    """starts a DistributedEvaluator in slave mode."""
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
    de = neat.DistributedEvaluator(
        addr,
        authkey=authkey,
        eval_function=eval_dummy_genome_nn,
        mode=MODE_SLAVE,
        num_workers=num_workers,
        )
    try:
        de.start(slave_wait=3, exit_on_stop=True)
    except SystemExit:
        pass
    else:
        raise Exception("DistributedEvaluator in slave mode did not try to exit!")
