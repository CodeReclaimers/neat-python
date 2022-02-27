"""tests for neat.distributed"""
from __future__ import print_function

import multiprocessing
import os
import platform
import random
import socket
import sys
import unittest

try:
    import threading
except ImportError:
    import dummy_threading as threading

    HAVE_THREADING = False
else:
    HAVE_THREADING = True

import neat
from neat.distributed import chunked, MODE_AUTO, MODE_PRIMARY, MODE_SECONDARY, ModeError, _STATE_RUNNING

ON_PYPY = platform.python_implementation().upper().startswith("PYPY")


def eval_dummy_genome_nn(genome, config):
    """dummy evaluation function"""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return 0.0


def test_chunked():
    """Test for neat.distributed.chunked()"""
    # test chunked(range(110), 10)
    # => 11 chunks of 10 elements
    d110 = list(range(110))
    d110c = chunked(d110, 10)
    assert len(d110c) == 11, "chunked(range(110), 10) created {0:n} chunks, not 11 chunks!".format(len(d110c))
    assert len(d110c[
                   0]) == 10, "chunked(range(110), 10) did not create chunks of length 10 (first chunk len is {0:n})!".format(
        len(d110c[0]))
    rec = []
    for e in d110c:
        rec += e
    assert rec == d110, "chunked(range(110), 10) created incorrect chunks ({0!r} vs expected {1!r})".format(rec, d110)
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
    assert len(d13c) == 5, "chunked(range(13), 3) created {0:n} chunks, not 5!".format(len(d13c))
    assert len(
        d13c[0]) == 3, "chunked(range(13), 3) did not create chunks of length 3 (first chunk len is {0:n})!".format(
        len(d13c[0]))
    assert len(d13c[-1]) == 1, "chunked(range(13), 3) created a last chunk of length {0:n}, not 1!".format(
        len(d13c[-1]))
    rec = []
    for e in d13c:
        rec += e
    assert rec == d13, "chunked(range(13), 3) created incorrect chunks ({0!r} vs expected {1!r})".format(rec, d13)


def test_host_is_local():
    """test for neat.distributed.host_is_local()"""
    tests = (
        # (hostname or ip, expected value)
        (b"localhost", True),
        (b"0.0.0.0", True),
        (b"127.0.0.1", True),
        (socket.gethostname(), True),
        (socket.getfqdn(), True),
        (b"github.com", False),
        (b"google.de", False),
        )
    for hostname, islocal in tests:
        try:
            result = neat.host_is_local(hostname)
        except EnvironmentError:  # give more feedback
            print("test_host_is_local: Error with hostname {0!r} (expected {1!r})".format(hostname,
                                                                                          islocal))
            raise
        else: # if do not want to do 'raise' above for some cases
            assert result == islocal, "Unexpected results for hostname/IP: {h}; Expected: {e}; Got: {r!r}".format(
                h=hostname, e=islocal, r=result)


def test_DistributedEvaluator_mode():
    """Tests for the mode determination of neat.distributed.DistributedEvaluator()"""
    # test auto mode setting
    # we also test that the mode is not
    # automatically determined when explicitly given.
    tests = (
        # (hostname or ip, mode to pass, expected mode)
        (b"localhost", MODE_PRIMARY, MODE_PRIMARY),
        (b"0.0.0.0", MODE_PRIMARY, MODE_PRIMARY),
        (b"localhost", MODE_SECONDARY, MODE_SECONDARY),
        (b"example.org", MODE_PRIMARY, MODE_PRIMARY),
        (socket.gethostname().encode("ascii"), MODE_SECONDARY, MODE_SECONDARY),
        (b"localhost", MODE_AUTO, MODE_PRIMARY),
        (socket.gethostname().encode("ascii"), MODE_AUTO, MODE_PRIMARY),
        (socket.getfqdn().encode("ascii"), MODE_AUTO, MODE_PRIMARY),
        (b"example.org", MODE_AUTO, MODE_SECONDARY),
        )
    for hostname, mode, expected in tests:
        for addr in ((hostname, 8022), hostname):
            try:
                de = neat.DistributedEvaluator(
                    addr,
                    authkey="abcd1234",
                    eval_function=eval_dummy_genome_nn,
                    mode=mode,
                    )
            except EnvironmentError:
                print("test_DistributedEvaluator_mode(): Error with hostname " +
                      "{!r}".format(hostname))
                raise
            result = de.mode
            assert result == expected, "Mode determination failed! Hostname: {h}; expected: {e}; got: {r!r}!".format(
                h=hostname, e=expected, r=result)

            if result == MODE_AUTO:
                raise Exception(
                    "DistributedEvaluator.__init__(mode=MODE_AUTO) did not automatically determine its mode!"
                    )
            elif (result == MODE_PRIMARY) and (not de.is_primary()):
                raise Exception(
                    "DistributedEvaluator.is_primary() returns False even if the evaluator is in primary mode!"
                    )
            elif (result == MODE_SECONDARY) and de.is_primary():
                raise Exception(
                    "DistributedEvaluator.is_primary() returns True even if the evaluator is in secondary mode!"
                    )
    # test invalid mode error
    try:
        de = neat.DistributedEvaluator(
            addr,
            authkey=u"abcd1234",
            eval_function=eval_dummy_genome_nn,
            mode="#invalid MODE!",
        )
        de.start()
    except ValueError:
        pass
    else:
        raise Exception("Passing an invalid mode did not cause an exception to be raised on start()!")


def test_json_bytes_dumps_loads():
    """
    Test for json_bytes_dumps() and json_bytes_loads().
    """
    test_objs = [
    1,
    2,
    3,
    [1, 2, 3],
    {"one": 1, "two": 2, "three": "three"},
    {"1": 2, "3":4, "5":6},
    u"unicode_string",
    "native_string",
    ]
    bytestype = type(b"some bytestring")
    for obj in test_objs:
        dumped = neat.distributed.json_bytes_dumps(obj)
        if type(dumped) != bytestype:
            raise Exception("neat.distributed.json_bytes_dumps({o}) did not return a bytestring!".format(o=repr(obj)))
        loaded = neat.distributed.json_bytes_loads(dumped)
        if loaded != obj:
            raise Exception("neat.distributed.json_bytes_loads(): {lo}; expected: {o}!".format(lo=repr(loaded), o=repr(obj)))


def test_DistributedEvaluator_primary_restrictions():
    """Test that primary-exclusive methods fail when called by the secondary nodes"""
    secondary = neat.DistributedEvaluator(
        (b"localhost", 8022),
        authkey=u"abcd1234",
        eval_function=eval_dummy_genome_nn,
        mode=MODE_SECONDARY,
    )
    try:
        secondary.stop()
    except ModeError:
        # only ignore ModeErrors
        # a RuntimeError should only be raised when in primary mode.
        pass
    else:
        raise Exception("A DistributedEvaluator in secondary mode could call stop()!")
    try:
        secondary.evaluate(None, None)  # we do not need valid values for this test
    except ModeError:
        # only ignore ModeErrors
        # other errors should only be raised when in primary mode.
        pass
    else:
        raise Exception("A DistributedEvaluator in secondary mode could call evaluate()!")


def test_distributed_evaluation_multiprocessing(do_mwcp=True):
    """
    Full test run using the Distributed Evaluator (fake nodes using processes).
    Note that this is not a very good test for the
    DistributedEvaluator, because we still work on
    one machine, not across multiple machines.
    We emulate the other machines using subprocesses
    created using the multiprocessing module.
    """
    addr = (b"localhost", random.randint(12000, 30000))
    authkey = u"abcd1234"
    mp = multiprocessing.Process(
        name="Primary evaluation process",
        target=run_primary,
        args=(addr, authkey, 19),  # 19 because stagnation is at 20
    )
    mp.start()
    if do_mwcp:
        mwcp = multiprocessing.Process(
            name="Child evaluation process (multiple workers)",
            target=run_secondary,
            args=(addr, authkey, 2),
        )
    swcp = multiprocessing.Process(
        name="Child evaluation process (direct evaluation)",
        target=run_secondary,
        args=(addr, authkey, 1),
    )
    swcp.daemon = True  # we cannot set this on mwcp
    if do_mwcp:
        mwcp.start()
    else:
        print("[distributed] Tests/Process-control: Skipping start of the multiworker secondary node process...")
    swcp.start()
    try:
        print("[distributed] Tests/Process-control: Joining process of primary node...")
        sys.stdout.flush()
        mp.join()
        print("[distributed] Tests/Process-control: successfully joined process of primary node.")
        if mp.exitcode != 0:
            raise Exception("Primary-node-process exited with status {s}!".format(s=mp.exitcode))
        if do_mwcp:
            if not mwcp.is_alive():
                print("[distributed] Tests/Process-control: mwcp is not 'alive'!")
            print("[distributed] Tests/Process-control: children: {c}.".format(c=multiprocessing.active_children()))
            print("[distributed] Tests/Process-control: Joining multiworker-secondary process...")
            sys.stdout.flush()
            mwcp.join()
            print("[distributed] Tests/Process-control: successfully joinded multiworker-secondary process.")
            if mwcp.exitcode != 0:
                raise Exception("Multiworker-secondary-process exited with status {s}!".format(s=mwcp.exitcode))
        if not swcp.is_alive():
            print("[distributed] Tests/Process-control: swcp is not 'alive'!")
        print("[distributed] Tests/Process-control: Joining singleworker-secondary process.")
        sys.stdout.flush()
        swcp.join()
        print("[distributed] Tests/Process-control: successfully joined singleworker-secondary process.")
        if swcp.exitcode != 0:
            raise Exception("Singleworker-secondary-process exited with status {s}!".format(s=swcp.exitcode))

    finally:
        if mp.is_alive():
            mp.terminate()
        if do_mwcp and mwcp.is_alive():
            mwcp.terminate()
        if swcp.is_alive():
            swcp.terminate()


def test_distributed_evaluation_invalid_authkey_multiprocessing(pc=multiprocessing.Process):
    """Test for DistributedEvaluator-behavior on invalid authkey (Fake nodes using processes)"""
    addr = (b"localhost", random.randint(12000, 30000))
    valid_authkey = u"Linux>Windows"
    invalid_authkey = u"Windows>Linux"
    print("[distributed] Tests/Process-control: starting primary node/thread...")
    mp = pc(
        name="Primary evaluation process",
        target=run_primary,
        args=(addr, valid_authkey, 19), # 19 because stagnation is at 20
        )
    mp.daemon = True
    mp.start()
    print("[distributed] Tests/Process-control: running secondary node in main process/thread...")
    try:
        run_secondary(addr, authkey=invalid_authkey, num_workers=1)
    except neat.distributed.AuthError:
        # expected
        print("[distributed] Tests/Process-control: catched expected AuthError.")
        pass
    else:
        print("[distributed] Tests/Process-control: did not catch expected AuthError! This is an Error!")
        raise Exception("Expected an auth error!")
    finally:
        print("[distributed] Tests/Process-control: Terminating primary process/thread.")
        if hasattr(mp, "terminate"):
            mp.terminate()


def test_distributed_evaluation_invalid_authkey_threaded():
    """Test for DistributedEvaluator-behavior on invalid authkey (Fake nodes using threads)"""
    test_distributed_evaluation_invalid_authkey_multiprocessing(pc=threading.Thread)


def test_distributed_evaluation_threaded():
    """
    Full test run using the Distributed Evaluator (fake nodes using threads).
    Note that this is not a very good test for the
    DistributedEvaluator, because we still work on
    one machine, not across multiple machines.
    We emulate the other machines using threads.
    This test is like test_distributed_evaluation_multiprocessing,
    but uses threads instead of processes.
    We use this to get better coverage.
    """
    if not HAVE_THREADING:
        raise unittest.SkipTest("Platform does not have threading")
    addr = (b"localhost", random.randint(12000, 30000))
    authkey = u"abcd1234"
    mp = threading.Thread(
        name="Primary evaluation thread",
        target=run_primary,
        args=(addr, authkey, 19),  # 19 because stagnation is set at 20
    )
    mp.start()
    mwcp = threading.Thread(
        name="Child evaluation thread (multiple workers)",
        target=run_secondary,
        args=(addr, authkey, 2),
    )
    swcp = threading.Thread(
        name="Child evaluation thread (direct evaluation)",
        target=run_secondary,
        args=(addr, authkey, 1),
    )
    swcp.daemon = True  # we cannot set this on mwcp
    mwcp.start()
    swcp.start()
    mp.join()
    mwcp.join()
    swcp.join()

    # we cannot check for exceptions in the threads.
    # however, these checks are also done in
    # test_distributed_evaluationmultiprocessing,
    # so they should not fail here.
    # also, this test is mainly for the coverage.


def run_primary(addr, authkey, generations):
    """Starts a DistributedEvaluator in primary mode."""
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
    p.add_reporter(neat.Checkpointer(max(1, int(generations / 3)), 5))

    # Run for the specified number of generations.
    de = neat.DistributedEvaluator(
        addr,
        authkey=authkey,
        eval_function=eval_dummy_genome_nn,
        mode=MODE_PRIMARY,
        secondary_chunksize=15,
        )
    print("[distributed] primary: starting DistributedEvaluator")
    sys.stdout.flush()
    de.start()
    print("[distributed] primary: starting evaluation of genomes...")
    sys.stdout.flush()
    p.run(de.evaluate, generations)
    print("[distributed] primary: evaluation finished.")
    sys.stdout.flush()
    de.stop(wait=5)
    print("[distributed] primary: did de.stop")
    sys.stdout.flush()

    stats.save()


def run_secondary(addr, authkey, num_workers=1):
    """Starts a DistributedEvaluator in secondary mode."""
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

    # Run for the specified number of generations.
    de = neat.DistributedEvaluator(
        addr,
        authkey=authkey,
        eval_function=eval_dummy_genome_nn,
        mode=MODE_SECONDARY,
        num_workers=num_workers,
        )
    print("[distributed] secondary: starting DistributedEvaluator...")
    sys.stdout.flush()
    try:
        de.start(secondary_wait=3, exit_on_stop=True, reconnect=False)
    except SystemExit:
        print("[distributed] secondary: caught expected SystemExit.")
        sys.stdout.flush()
        pass
    else:
        print("[distributed] secondary: expected a SystemExit; not SystemExit caught!")
        sys.stdout.flush()
        raise Exception("DistributedEvaluator in secondary mode did not try to exit!")


if __name__ == '__main__':
    test_chunked()
    test_host_is_local()
    test_DistributedEvaluator_mode()
    test_DistributedEvaluator_primary_restrictions()
    test_distributed_evaluation_multiprocessing(do_mwcp=True)
    if HAVE_THREADING:
        test_distributed_evaluation_threaded()
