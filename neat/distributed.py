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

NOTE:
    This module is in a **beta** state, and still *unstable* even in single-machine testing. Reliability is likely to vary, including depending on the Python version
    and implementation (e.g., cpython vs pypy) in use and the likelihoods of timeouts (due to machine and/or network slowness). In particular, while the code can try
    to reconnect between between primary and secondary nodes, as noted in the `multiprocessing` documentation this may not work due to data loss/corruption. Note also
    that this module is not responsible for starting the script copies on the different compute nodes, since this is very site/configuration-dependent.


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
from multiprocessing import managers
from argparse import Namespace

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
MODE_PRIMARY = MODE_MASTER = 1  # enforce primary mode
MODE_SECONDARY = MODE_SLAVE = 2  # enforce secondary mode

# states to determine whether the secondaries should shut down
_STATE_RUNNING = 0
_STATE_SHUTDOWN = 1
_STATE_FORCED_SHUTDOWN = 2


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


def _determine_mode(addr, mode):
    """
    Returns the mode which should be used.
    If mode is MODE_AUTO, this is determined by checking if 'addr' points to the
    local host. If it does, return MODE_PRIMARY, else return MODE_SECONDARY.
    If mode is either MODE_PRIMARY or MODE_SECONDARY,
    return the 'mode' argument. Otherwise, a ValueError is raised.
    """
    if isinstance(addr, tuple):
        host = addr[0]
    elif type(addr) == type(b"binary_string"):
        host = addr
    else:
        raise TypeError("'addr' needs to be a tuple or an bytestring!")
    if mode == MODE_AUTO:
        if host_is_local(host):
            return MODE_PRIMARY

        return MODE_SECONDARY
    elif mode in (MODE_SECONDARY, MODE_PRIMARY):
        return mode
    else:
        raise ValueError("Invalid mode {!r}!".format(mode))


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


class _ExtendedManager(object):
    """A class for managing the multiprocessing.managers.SyncManager"""
    __safe_for_unpickling__ = True  # this may not be safe for unpickling,
                                    # but this is required by pickle.

    def __init__(self, addr, authkey, mode, start=False):
        self.addr = addr
        self.authkey = authkey
        self.mode = _determine_mode(addr, mode)
        self.manager = None
        self._secondary_state= multiprocessing.managers.Value(int, _STATE_RUNNING)
        if start:
            self.start()

    def __reduce__(self):
        """
        This method is used by pickle to serialize instances of this class.
        """
        return (
            self.__class__,
            (self.addr, self.authkey, self.mode, True),
            )

    def start(self):
        """Starts or connects to the manager."""
        if self.mode == MODE_PRIMARY:
            i = self._start()
        else:
            i = self._connect()
        self.manager = i

    def stop(self):
        """Stops the manager."""
        self.manager.shutdown()

    def set_secondary_state(self, value):
        """Sets the value for 'secondary_state'."""
        if value not in (_STATE_RUNNING, _STATE_SHUTDOWN, _STATE_FORCED_SHUTDOWN):
            raise ValueError(
                "State {!r} is invalid - needs to be one of _STATE_RUNNING, _STATE_SHUTDOWN, or _STATE_FORCED_SHUTDOWN".format(
                    value)
                )
        if self.manager is None:
            raise RuntimeError("Manager not started")
        self.manager.set_state(value)

    def _get_secondary_state(self):
        """
        Returns the value for 'secondary_state'.
        This is required for the manager.
        """
        return self._secondary_state

    def _get_manager_class(self, register_callables=False):
        """
        Returns a new 'Manager' subclass with registered methods.
        If 'register_callable' is True, defines the 'callable' arguments.
        """

        class _EvaluatorSyncManager(managers.BaseManager):
            """
            A custom BaseManager.
            Please see the documentation of `multiprocessing` for more
            information.
            """
            pass

        inqueue = queue.Queue()
        outqueue = queue.Queue()
        namespace = Namespace()

        if register_callables:
            _EvaluatorSyncManager.register(
                "get_inqueue",
                callable=lambda: inqueue,
                )
            _EvaluatorSyncManager.register(
                "get_outqueue",
                callable=lambda: outqueue,
                )
            _EvaluatorSyncManager.register(
                "get_state",
                callable=self._get_secondary_state,
                )
            _EvaluatorSyncManager.register(
                "set_state",
                callable=lambda v: self._secondary_state.set(v),
                )
            _EvaluatorSyncManager.register(
                "get_namespace",
                callable=lambda: namespace,
                )


        else:
            _EvaluatorSyncManager.register(
                "get_inqueue",
                )
            _EvaluatorSyncManager.register(
                "get_outqueue",
                )
            _EvaluatorSyncManager.register(
                "get_state",
                )
            _EvaluatorSyncManager.register(
                "set_state",
                )
            _EvaluatorSyncManager.register(
                "get_namespace",
                )
        return _EvaluatorSyncManager

    def _connect(self):
        """Connects to the manager."""
        cls = self._get_manager_class(register_callables=False)
        ins = cls(address=self.addr, authkey=self.authkey)
        ins.connect()
        return ins

    def _start(self):
        """Starts the manager."""
        cls = self._get_manager_class(register_callables=True)
        ins = cls(address=self.addr, authkey=self.authkey)
        ins.start()
        return ins

    @property
    def secondary_state(self):
        """Whether the secondary nodes should still process elements."""
        v = self.manager.get_state()
        return v.get()

    def get_inqueue(self):
        """Returns the inqueue."""
        if self.manager is None:
            raise RuntimeError("Manager not started")
        return self.manager.get_inqueue()

    def get_outqueue(self):
        """Returns the outqueue."""
        if self.manager is None:
            raise RuntimeError("Manager not started")
        return self.manager.get_outqueue()

    def get_namespace(self):
        """Returns the namespace."""
        if self.manager is None:
            raise RuntimeError("Manager not started")
        return self.manager.get_namespace()


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
            except (RuntimeError, AttributeError): # pragma: no cover
                print("multiprocessing.cpu_count() gave an error; assuming 1",
                      file=sys.stderr)
                self.num_workers = 1
        self.worker_timeout = worker_timeout
        self.mode = _determine_mode(self.addr, mode)
        self.em = _ExtendedManager(self.addr, self.authkey, mode=self.mode, start=False)
        self.inqueue = None
        self.outqueue = None
        self.namespace = None
        self.started = False

    def __getstate__(self):
        """Required by the pickle protocol."""
        # we do not actually save any state, but we need __getstate__ to be
        # called.
        return True  # return some nonzero value

    def __setstate__(self, state):
        """Called when instances of this class are unpickled."""
        self._set_shared_instances()

    def is_primary(self):
        """Returns True if the caller is the primary node"""
        return (self.mode == MODE_PRIMARY)

    def is_master(self): # pragma: no cover
        """Returns True if the caller is the primary (master) node"""
        warnings.warn("Use is_primary, not is_master", DeprecationWarning)
        return self.is_primary()

    def start(self, exit_on_stop=True, secondary_wait=0, reconnect=False):
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
        If 'reconnect' is True, the secondary nodes will try to reconnect when
        the connection is lost. In this case, sys.exit() will only be called
        when 'exit_on_stop' is True and the primary node send a forced shutdown
        command.
        """
        if self.started:
            raise RuntimeError("DistributedEvaluator already started!")
        self.started = True
        if self.mode == MODE_PRIMARY:
            self._start_primary()
        elif self.mode == MODE_SECONDARY:
            time.sleep(secondary_wait)
            self._start_secondary()
            self._secondary_loop(reconnect=reconnect)
            if exit_on_stop:
                sys.exit(0)
        else:
            raise ValueError("Invalid mode {!r}!".format(self.mode))

    def stop(self, wait=1, shutdown=True, force_secondary_shutdown=False):
        """
        Stops all secondaries.
        'wait' specifies the time (in seconds) to wait before shutting down the
        manager or returning.
        If 'shutdown', shutdown the manager.
        If 'force_secondary_shutdown', shutdown the secondary nodes even if
        they are started with 'reconnect=True'.
        """
        if self.mode != MODE_PRIMARY:
            raise ModeError("Not in primary mode!")
        if not self.started:
            raise RuntimeError("Not yet started!")
        if force_secondary_shutdown:
            state = _STATE_FORCED_SHUTDOWN
        else:
            state = _STATE_SHUTDOWN
        self.em.set_secondary_state(state)
        time.sleep(wait)
        if shutdown:
            self.em.stop()
        self.started = False
        self.inqueue = self.outqueue = self.namespace = None

    def _start_primary(self):
        """Start as the primary"""
        self.em.start()
        self.em.set_secondary_state(_STATE_RUNNING)
        self._set_shared_instances()

    def _start_secondary(self):
        """Start as a secondary."""
        self.em.start()
        self._set_shared_instances()

    def _set_shared_instances(self):
        """Sets attributes from the shared instances."""
        self.inqueue = self.em.get_inqueue()
        self.outqueue = self.em.get_outqueue()
        self.namespace = self.em.get_namespace()

    def _reset_em(self):
        """Resets self.em and the shared instances."""
        self.em = _ExtendedManager(self.addr, self.authkey, mode=self.mode, start=False)
        self.em.start()
        self._set_shared_instances()

    def _secondary_loop(self, reconnect=False):
        """The worker loop for the secondary nodes."""
        if self.num_workers > 1:
            pool = multiprocessing.Pool(self.num_workers)
        else:
            pool = None
        should_reconnect = True
        while should_reconnect:
            i = 0
            running = True
            try:
                self._reset_em()
            except (socket.error, EOFError, IOError, OSError, socket.gaierror, TypeError):
                continue
            while running:
                i += 1
                if i % 5 == 0:
                    # for better performance, only check every 5 cycles
                    try:
                        state = self.em.secondary_state
                    except (socket.error, EOFError, IOError, OSError, socket.gaierror, TypeError):
                        if not reconnect:
                            raise
                        else:
                            break
                    if state == _STATE_FORCED_SHUTDOWN:
                        running = False
                        should_reconnect = False
                    elif state == _STATE_SHUTDOWN:
                        running = False
                    if not running:
                        continue
                try:
                    tasks = self.inqueue.get(block=True, timeout=0.2)
                except queue.Empty:
                    continue
                except (socket.error, EOFError, IOError, OSError, socket.gaierror, TypeError):
                    break
                except (managers.RemoteError, multiprocessing.ProcessError) as e:
                    if ('Empty' in repr(e)) or ('TimeoutError' in repr(e)):
                        continue
                    if (('EOFError' in repr(e)) or ('PipeError' in repr(e)) or
                        ('AuthenticationError' in repr(e))): # Second for Python 3.X, Third for 3.6+
                        break
                    raise
                if pool is None:
                    res = []
                    for genome_id, genome, config in tasks:
                        fitness = self.eval_function(genome, config)
                        res.append((genome_id, fitness))
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
                    res = zip(genome_ids, results)
                try:
                    self.outqueue.put(res)
                except (socket.error, EOFError, IOError, OSError, socket.gaierror, TypeError):
                    break
                except (managers.RemoteError, multiprocessing.ProcessError) as e:
                    if ('Empty' in repr(e)) or ('TimeoutError' in repr(e)):
                        continue
                    if (('EOFError' in repr(e)) or ('PipeError' in repr(e)) or
                        ('AuthenticationError' in repr(e))): # Second for Python 3.X, Third for 3.6+
                        break
                    raise

            if not reconnect:
                should_reconnect = False
                break
        if pool is not None:
            pool.terminate()

    def evaluate(self, genomes, config):
        """
        Evaluates the genomes.
        This method raises a ModeError if the
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
        while len(tresults) < n_tasks:
            try:
                sr = self.outqueue.get(block=True, timeout=0.2)
            except (queue.Empty, managers.RemoteError):
                continue
            tresults.append(sr)
        results = []
        for sr in tresults:
            results += sr
        for genome_id, fitness in results:
            genome = id2genome[genome_id]
            genome.fitness = fitness
