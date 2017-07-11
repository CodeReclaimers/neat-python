"""distributed evaluation of genomes"""
import multiprocessing
import socket
import sys
import threading
import time

try:
    import Queue as queue
except ImportError:
    import queue

from multiprocessing import managers
from argparse import Namespace


# Some of this code is based on
# http://eli.thegreenplace.net/2012/01/24/distributed-computing-in-python-with-multiprocessing
# According to the website, the code is in the public domain
# ('public domain' links to unlicense.org).
# This means that we can use the code from this website.
# Thanks to Eli Bendersky for making his code open source.


# modes to determine the role of a machine
# the master handles the evolution of the genomes
# the slave handles the evaluation of the genomes
MODE_AUTO = 0  # auto determine server role
MODE_MASTER = 1  # enforce master mode
MODE_SLAVE = 2  # enforce slave mode


class RoleError(RuntimeError):
    """
    An exception raised when a role-specific method is being
    called without being in the role.
    """


def host_is_local(hostname, port=None):
    """
    Returns True if the hostname points to the localhost, otherwise False.
    """
    if port is None:
        port = 22  # no port specified, lets just use the ssh port
    hostname = socket.getfqdn(hostname)
    if hostname in ("localhost", "0.0.0.0"):
        return True
    localhost = socket.gethostname()
    localaddrs = socket.getaddrinfo(localhost, port)
    targetaddrs = socket.getaddrinfo(hostname, port)
    for (family, socktype, proto, canonname, sockaddr) in localaddrs:
        for (rfamily, rsocktype, rproto, rcanonname, rsockaddr) in targetaddrs:
            if rsockaddr[0] == sockaddr[0]:
                return True
    return False


def chunked(data, chunksize):
    """
    Returns a list of chunks containing at most 'chunksize' elements of data.
    """
    if chunksize < 1:
        raise ValueError("Chunksize must be at least 1!")
    if not isinstance(chunksize, int):
        raise ValueError("Chunksize needs to be an integer")
    res = []
    cur = []
    for e in data:
        cur.append(e)
        if len(cur) >= chunksize:
            res.append(cur)
            cur = []
    if len(cur) > 0:
        res.append(cur)
    return res


class DistributedEvaluator(object):
    """An evaluator working across multiple machines"""
    def __init__(
            self,
            addr,
            authkey,
            eval_function,
            slave_chunksize=1,
            num_workers=None,
            worker_timeout=60,
            mode=MODE_AUTO
            ):
        """
        'addr' should be a tuple of (hostname, port) pointing to the machine
        running the DistributedEvaluator in master mode. If mode is MODE_AUTO,
        the mode is determined by checking wether the hostname points to this
        host or not.
        'authkey' is the password used to restrict access to the manager.
        All DistributedEvaluators need to use the same authkey.
        'eval_function' should take two arguments (a genome object and the
        configuration) and return a single float (the genome's fitness).
        'slave_chunksize' specifies the number of genomes which will be send to
        a slave at once.
        'num_workers' is the number of child processes to use if in client
        mode. It defaults to None, which means multiprocessing.cpu_count()
        is used to determine this value.
        If it equals to 1, do evaluate it in this process.
        'worker_timeout' specifies the timeout for getting the results from
        a worker when in slave mode.
        """
        self.addr = addr
        self.authkey = authkey
        self.eval_function = eval_function
        self.slave_chunksize = slave_chunksize
        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = num_workers
        self.worker_timeout = worker_timeout
        if mode == MODE_AUTO:
            if host_is_local(self.addr[0]):
                mode = MODE_MASTER
            else:
                mode = MODE_SLAVE
        self.mode = mode
        self.manager = None
        self.started = False

    def is_master(self):
        """returns True if the host is the master"""
        return (self.mode == MODE_MASTER)

    def start(self, exit_on_stop=True, slave_wait=0):
        """
        If the DistributedEvaluator is in master mode, start the manager
        process and returns. In this case, the 'exit_on_stop' argument will
        be ignored.
        If the DistributedEvaluator is in slave mode, connect to the manager
        and and wait for tasks.
        If in slave mode and 'exit_on_stop' is True, sys.exit() will be called
        when the connection is lost.
        'slave_wait' specifies the time (in seconds) to sleep before actually
        starting when in client mode.
        """
        if self.started:
            raise RuntimeError("DistributedEvaluator already started!")
        self.started = True
        if self.mode == MODE_MASTER:
            self._start_master()
        elif self.mode == MODE_SLAVE:
            time.sleep(slave_wait)
            self._start_slave()
            self._slave_loop()
            if exit_on_stop:
                sys.exit(0)
        else:
            raise ValueError("Invalid mode!")

    def stop(self, wait=1):
        """stops all slaves."""
        if self.mode != MODE_MASTER:
            raise RoleError("Not in master mode!")
        if not self.started:
            raise RuntimeError("Not yet started!")
        stopevent = self.manager.get_stopevent()
        stopevent.set()
        time.sleep(wait)

    def _start_master(self):
        """starts as the master"""
        inqueue = queue.Queue()
        outqueue = queue.Queue()
        namespace = Namespace()
        stop_event = threading.Event()

        class _EvaluatorSyncManager(managers.SyncManager):
            """
            A custom SyncManager.
            Please see the documentation of multiprocesing for more
            informations.
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
            "get_namespace",
            callable=lambda: namespace,
            )
        _EvaluatorSyncManager.register(
            "get_stopevent",
            callable=lambda: stop_event,
            )

        self.manager = _EvaluatorSyncManager(
            address=self.addr,
            authkey=self.authkey,
            )
        self.manager.start()

        self.inqueue = self.manager.get_inqueue()
        self.outqueue = self.manager.get_outqueue()

    def _start_slave(self):
        """starts as a slave"""

        class _EvaluatorSyncManager(managers.SyncManager):
            """
            A custom SyncManager.
            Please see the documentation of multiprocesing for more
            informations.
            """
            pass

        _EvaluatorSyncManager.register("get_inqueue")
        _EvaluatorSyncManager.register("get_outqueue")
        _EvaluatorSyncManager.register("get_namespace")
        _EvaluatorSyncManager.register("get_stopevent")

        self.manager = _EvaluatorSyncManager(
            address=self.addr,
            authkey=self.authkey,
            )
        self.manager.connect()

    def _slave_loop(self):
        """the worker loop for the slave"""
        inqueue = self.manager.get_inqueue()
        outqueue = self.manager.get_outqueue()
        stopevent = self.manager.get_stopevent()
        if self.num_workers > 1:
            pool = multiprocessing.Pool(self.num_workers)
        else:
            pool = None
        while not stopevent.is_set():
            try:
                tasks = inqueue.get(block=True, timeout=0.2)
            except (managers.RemoteError, queue.Empty):
                continue
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
        This method raises an RoleError when this
        DistributedEvaluator is not in master mode.
        """
        if self.mode != MODE_MASTER:
            raise RoleError("Not in master mode!")
        stopevent = self.manager.get_stopevent()
        tasks = [(genome_id, genome, config) for genome_id, genome in genomes]
        id2genome = {genome_id: genome for genome_id, genome in genomes}
        tasks = chunked(tasks, self.slave_chunksize)
        n_tasks = len(tasks)
        for task in tasks:
            self.inqueue.put(task)
        tresults = []
        while len(tresults) < n_tasks:
            try:
                sr = self.outqueue.get(block=True, timeout=0.2)
            except (queue.Empty, managers.RemoteError):
                if stopevent.is_set():
                    raise SystemExit("Received stop event.")
                continue
            tresults.append(sr)
        results = []
        for sr in tresults:
            results += sr
        for genome_id, fitness in results:
            genome = id2genome[genome_id]
            genome.fitness = fitness
