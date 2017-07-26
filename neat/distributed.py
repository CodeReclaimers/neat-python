"""
Distributed evaluation of genomes.

About compute nodes:
The primary node (=the node which creates and mutates genomes) and the secondary
nodes (=the nodes which evaluate genomes) can execute the same script. The
role of a compute node is determined using the ``mode`` argument of the
DistributedEvaluator. If the mode is MODE_AUTO, the `host_is_local()` function
is used to check if the ``addr`` argument points to the localhost. If it does,
the compute node starts as a primary node, otherwise as a secondary node. If
``mode`` is MODE_PRIMARY, the compute node always starts as a primary node. If
``mode`` is MODE_SECONDARY, the compute node will always start as a secondary node.

There can only be one primary node per NEAT, but any number of secondary nodes.
The primary node will not evaluate any genomes, which means you will always need
at least two compute nodes.

You can run any number of compute nodes on the same physical machine (or VM).
However, if a machine has both a primary node and one or more secondary nodes,
MODE_AUTO cannot be used for those secondary nodes - MODE_SECONDARY will need to be
specified.

Usage:
1. Import modules and define the evaluation logic (the eval_genome function).
  (After this, check for ``if __name__ == '__main__'``, and put the rest of
  the code inside the body of the statement.)
2. Load config and create a population - here, the variable ``p``.
3. If required, create and add reporters.
4. Create a ``DistributedEvaluator(addr_of_primary_node, b'some_password',
  eval_function, mode=MODE_AUTO)`` - here, the variable ``de``.
5. Call ``de.start(exit_on_stop=True)``. The `start()` call will block on the
  secondary nodes and call `sys.exit(0)` when the NEAT evolution finishes. This
  means that the following code will only be executed on the primary node.
6. Start the evaluation using ``p.run(de.evaluate, number_of_generations)``.
7. Stop the secondary nodes using ``de.stop()``.
8. You are done. You may want to save the winning genome or show some statistics.

See ``examples/xor/evolve-feedforward-distributed.py`` for a complete example.

Utility functions:

``host_is_local(hostname, port=22)`` returns True if ``hostname`` points to
the local node/host. This can be used to check if a compute node will run as
a primary node or as a secondary node with MODE_AUTO.

``chunked(data, chunksize)``: splits data into a list of chunks with at most
``chunksize`` elements.
"""
from __future__ import print_function

import socket
import sys
import time
import warnings

# below still needed for queue.Empty
try:
    # pylint: disable=import-error
    import Queue as queue
except ImportError:
    # pylint: disable=import-error
    import queue

import multiprocessing
from multiprocessing import managers, Queue
from argparse import Namespace # why not from SyncManager?

from ctypes import c_bool

# Some of this code is based on
# http://eli.thegreenplace.net/2012/01/24/distributed-computing-in-python-with-multiprocessing
# According to the website, the code is in the public domain
# ('public domain' links to unlicense.org).
# This means that we can use the code from this website.
# Thanks to Eli Bendersky for making his code open for use.


# modes to determine the role of a compute node
# the primary handles the evolution of the genomes
# the secondary handles the evaluation of the genomes
MODE_AUTO = 0  # auto-determine mode
MODE_PRIMARY = 1  # enforce primary mode
MODE_SECONDARY = 2  # enforce secondary mode
MODE_MASTER = MODE_PRIMARY # backward compatibility
MODE_SLAVE = MODE_SECONDARY # ditto

class ModeError(RuntimeError):
    """
    An exception raised when a mode-specific method is being
    called without being in the mode - either a primary-specific method
    called by a secondary node or a secondary-specific method called by a primary node.
    """
    pass


def host_is_local(hostname, port=22): # no port specified, just use the ssh port
    """
    Returns True if the hostname points to the localhost, otherwise False.
    """
    hostname = socket.getfqdn(hostname)
    if hostname in ("localhost", "0.0.0.0", "127.0.0.1", "1.0.0.127.in-addr.arpa",
                    "1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.ip6.arpa"):
        return True
    localhost = socket.gethostname()
    if hostname == localhost:
        return True
    localaddrs = socket.getaddrinfo(localhost, port)
    targetaddrs = socket.getaddrinfo(hostname, port)
    for (ignored_family, ignored_socktype, ignored_proto, ignored_canonname,
         sockaddr) in localaddrs:
        for (ignored_rfamily, ignored_rsocktype, ignored_rproto,
             ignored_rcanonname, rsockaddr) in targetaddrs:
            if rsockaddr[0] == sockaddr[0]:
                return True
    return False


def chunked(data, chunksize):
    """
    Returns a list of chunks containing at most ``chunksize`` elements of data.
    """
    if chunksize < 1:
        raise ValueError("Chunksize must be at least 1!")
    if int(chunksize) != chunksize:
        raise ValueError("Chunksize needs to be an integer")
    res = []
    cur = []
    for e in data:
        cur.append(e)
        if len(cur) >= chunksize:
            res.append(cur)
            cur = []
    if cur:
        res.append(cur)
    return res


class DistributedEvaluator(object):
    """An evaluator working across multiple machines"""
    def __init__(
            self,
            addr,
            authkey,
            eval_function,
            secondary_chunksize=1,
            num_workers=None,
            worker_timeout=60,
            mode=MODE_AUTO,
            ):
        """
        ``addr`` should be a tuple of (hostname, port) pointing to the machine
        running the DistributedEvaluator in primary mode. If mode is MODE_AUTO,
        the mode is determined by checking whether the hostname points to this
        host or not.
        ``authkey`` is the password used to restrict access to the manager; see
        ``Authentication Keys`` in the `multiprocessing` manual for more information.
        All DistributedEvaluators need to use the same authkey. Note that this needs
        to be a `bytes` object for Python 3.X, and should be in 2.7 for compatibility
        (identical in 2.7 to a `str` object).
        ``eval_function`` should take two arguments (a genome object and the
        configuration) and return a single float (the genome's fitness).
        'secondary_chunksize' specifies the number of genomes that will be sent to
        a secondary at any one time.
        ``num_workers`` is the number of child processes to use if in secondary
        mode. It defaults to None, which means `multiprocessing.cpu_count()`
        is used to determine this value. If 1 in a secondary node, the process creating
        the DistributedEvaluator instance will also do the evaulations.
        ``worker_timeout`` specifies the timeout (in seconds) for a secondary node
        getting the results from a worker subprocess; if None, there is no timeout.
        ``mode`` specifies the mode to run in; it defaults to MODE_AUTO.
        """
        self.addr = addr
        self.authkey = authkey
        self.eval_function = eval_function
        self.secondary_chunksize = secondary_chunksize
        self.slave_chunksize = secondary_chunksize # backward compatibility
        if num_workers:
            self.num_workers = num_workers
        else:
            try:
                self.num_workers = max(1,multiprocessing.cpu_count())
            except (RuntimeError, AttributeError):
                print("multiprocessing.cpu_count() gave an error; assuming 1",
                      file=sys.stderr)
                self.num_workers = 1
        self.worker_timeout = worker_timeout
        if mode == MODE_AUTO:
            if host_is_local(self.addr[0]):
                mode = MODE_PRIMARY
            else:
                mode = MODE_SECONDARY
        elif mode not in (MODE_SECONDARY, MODE_PRIMARY):
            raise ValueError("Invalid mode {!r}!".format(mode))
        self.mode = mode
        self.manager = None
        self.inqueue = None
        self.outqueue = None
        self.eventqueue = None
        self.namespace = None
        self.started = False
        self.do_stop = None
        self.saw_EOFError = False

    def is_primary(self):
        """Returns True if the caller is the primary node"""
        return (self.mode == MODE_PRIMARY)

    def is_master(self):
        """Returns True if the caller is the primary (master) node"""
        warnings.warn("Use is_primary, not is_master", DeprecationWarning)
        return self.is_primary()

    def start(self, exit_on_stop=True, secondary_wait=0):
        """
        If the DistributedEvaluator is in primary mode, starts the manager
        process and returns. In this case, the ``exit_on_stop`` argument will
        be ignored.
        If the DistributedEvaluator is in secondary mode, it connects to the manager
        and waits for tasks.
        If in secondary mode and ``exit_on_stop`` is True, sys.exit() will be called
        when the connection is lost.
        ``secondary_wait`` specifies the time (in seconds) to sleep before actually
        starting when in secondary mode.
        """
        if self.started:
            raise RuntimeError("DistributedEvaluator already started!")
        self.started = True
        if self.mode == MODE_PRIMARY:
            if self.do_stop:
                self.do_stop = None
            self._start_primary()
        elif self.mode == MODE_SECONDARY:
            time.sleep(secondary_wait)
            self._start_secondary()
            self._secondary_loop()
##            print("Finished secondary loop; self.do_stop is {0!r} ({1!r})".format(self.do_stop,
##                                                                                  self.do_stop.__dict__))
##            sys.stdout.flush()
            if exit_on_stop:
                sys.exit(0)
            self.saw_EOFError = False
            self.do_stop = None
        else:
            raise ValueError("Invalid mode {!r}!".format(self.mode))

    def stop(self, wait=1, shutdown=True):
        """Stops all secondaries."""
        if self.mode != MODE_PRIMARY:
            raise ModeError("Not in primary mode!")
        if not self.started:
            raise RuntimeError("Not yet started!")
##        print("Setting do_stop to True")
##        sys.stdout.flush()
        self.do_stop = True
        time.sleep(wait)
        if shutdown:
##            print("Shutting down manager")
##            sys.stdout.flush()
##            if self.inqueue:
##                self.inqueue.close()
##            if self.outqueue:
##                self.outqueue.close()
##            if self.eventqueue:
##                self.eventqueue.close()
            self.manager.shutdown()
        self.started = False
        if multiprocessing.active_children() == 0:
            self.do_stop = None

    def _start_primary(self):
        """Start as the primary"""
        inqueue = Queue()
        outqueue = Queue()
        eventqueue = Queue()
        namespace = Namespace()

        class _EvaluatorSyncManager(managers.SyncManager):
            """
            A custom SyncManager.
            Please see the documentation of `multiprocessing` for more
            information.
            """
            pass

        _EvaluatorSyncManager.register(
            "get_inqueue",
            callable=lambda: inqueue,
            )
        _EvaluatorSyncManager.register(
            "get_outqueue",
            callable=lambda: outqueue,
            )
        _EvaluatorSyncManager.register(
            "get_eventqueue",
            callable=lambda: eventqueue,
            )
        _EvaluatorSyncManager.register(
            "get_namespace",
            callable=lambda: namespace,
            )

        if self.manager: # making sure does properly
##            if self.inqueue:
##                self.inqueue.close()
##            if self.outqueue:
##                self.outqueue.close()
##            if self.eventqueue:
##                self.eventqueue.close()
            self.manager.shutdown()

        self.manager = _EvaluatorSyncManager(
            address=self.addr,
            authkey=self.authkey,
            )
        self.manager.start()

        self.inqueue = self.manager.get_inqueue()
        self.outqueue = self.manager.get_outqueue()
        self.eventqueue = self.manager.get_eventqueue()
        self.namespace = self.manager.get_namespace()

        self.do_stop = self.manager.Value(c_bool, False)
##        print("Putting do_stop on eventqueue")
##        sys.stdout.flush()
        eventqueue.put(self.do_stop)
##        print("Put do_stop on eventqueue")
##        sys.stdout.flush()


    def _start_secondary(self):
        """Start as a secondary."""

        class _EvaluatorSyncManager(managers.SyncManager):
            """
            A custom SyncManager.
            Please see the documentation of `multiprocessing` for more
            information.
            """
            pass

        _EvaluatorSyncManager.register("get_inqueue")
        _EvaluatorSyncManager.register("get_outqueue")
        _EvaluatorSyncManager.register("get_eventqueue")
        _EvaluatorSyncManager.register("get_namespace")

        # if already have manager, below should result in it getting garbage-collected

        self.manager = _EvaluatorSyncManager(
            address=self.addr,
            authkey=self.authkey,
            )
        self.manager.connect()

        self.saw_EOFError = False

        eventqueue = self.manager.get_eventqueue()
        while (self.do_stop is None) and not self.saw_EOFError:
            try:
                self.do_stop = eventqueue.get(block=True,timeout=0.2)
            except queue.Empty:
##                print("Blocked on eventqueue with queue.Empty")
##                sys.stdout.flush()
                continue
            except (EOFError, IOError):
                self.saw_EOFError = True
                break
            except (managers.RemoteError, multiprocessing.ProcessError) as e:
                if ('Empty' in repr(e)) or ('TimeoutError' in repr(e)):
##                    print("Blocked on eventqueue with {!r}".format(e))
##                    sys.stdout.flush()
                    continue
                if (('EOFError' in repr(e)) or ('PipeError' in repr(e)) or
                    ('AuthenticationError' in repr(e))): # Second for Python 3.X, Third for Python 3.6+
                    self.saw_EOFError = True
                    break
                raise

    def _secondary_loop(self):
        """The worker loop for the secondary"""
        if self.saw_EOFError:
            return

        inqueue = self.manager.get_inqueue()
        outqueue = self.manager.get_outqueue()
        namespace = self.manager.get_namespace()

        if self.num_workers > 1:
            pool = multiprocessing.Pool(self.num_workers)
        else:
            pool = None
        while not (self.do_stop._value or self.saw_EOFError):
            try:
                tasks = inqueue.get(block=True, timeout=0.2)
            except queue.Empty:
                continue
            except (EOFError, IOError):
                self.saw_EOFError = True
                break
            except (managers.RemoteError, multiprocessing.ProcessError) as e:
                if ('Empty' in repr(e)) or ('TimeoutError' in repr(e)):
                    continue
                if (('EOFError' in repr(e)) or ('PipeError' in repr(e)) or
                    ('AuthenticationError' in repr(e))): # Second for Python 3.X, Third for 3.6+
                    self.saw_EOFError = True
                    break
                raise
            if pool is None:
                res = []
                for genome_id, genome, config in tasks:
                    fitness = self.eval_function(genome, config)
                    res.append((genome_id, fitness))
                outqueue.put(res)
            else:
                genome_ids = []
                jobs = []
                for genome_id, genome, config in tasks:
                    genome_ids.append(genome_id)
                    jobs.append(
                        pool.apply_async(
                            self.eval_function, (genome, config)
                            )
                        )
                results = [
                    job.get(timeout=self.worker_timeout) for job in jobs
                    ]
                zipped = zip(genome_ids, results)
                outqueue.put(zipped)

    def evaluate(self, genomes, config):
        """
        Evaluates the genomes.
        This method raises a ModeError when this
        DistributedEvaluator is not in primary mode.
        """
        if self.mode != MODE_PRIMARY:
            raise ModeError("Not in primary mode!")
        tasks = [(genome_id, genome, config) for genome_id, genome in genomes]
        id2genome = {genome_id: genome for genome_id, genome in genomes}
        tasks = chunked(tasks, self.secondary_chunksize)
        n_tasks = len(tasks)
        for task in tasks:
            self.inqueue.put(task)
        tresults = []
##        print("Self.do_stop is {0!r} ({1!r})".format(self.do_stop,
##                                                     self.do_stop.__dict__))
##        sys.stdout.flush()
        while len(tresults) < n_tasks:
            try:
                sr = self.outqueue.get(block=True, timeout=0.2)
            except (queue.Empty, managers.RemoteError):
                if hasattr(self.do_stop, '_value') and getattr(self.do_stop, '_value'):
                    raise SystemExit("Received stop.")
                elif self.eventqueue.empty():
##                    print("Putting do_stop on eventqueue again")
##                    sys.stdout.flush()
                    self.eventqueue.put(self.do_stop)
##                    print("Put do_stop on eventqueue again")
##                    sys.stdout.flush()
                continue
            tresults.append(sr)
        results = []
        for sr in tresults:
            results += sr
        for genome_id, fitness in results:
            genome = id2genome[genome_id]
            genome.fitness = fitness
##        print("Finished with de.evaluate")
##        sys.stdout.flush()
