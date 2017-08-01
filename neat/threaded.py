"""Threaded evaluation of genomes"""
from __future__ import print_function

import warnings

try:
    import threading
except ImportError: # pragma: no cover
    import dummy_threading as threading
    HAVE_THREADS = False
else:
    HAVE_THREADS = True

try:
    # pylint: disable=import-error
    import Queue as queue
except ImportError:
    # pylint: disable=import-error
    import queue

class ThreadedEvaluator(object):
    """
    A threaded genome evaluator.
    Useful on python implementations without GIL (Global Interpreter Lock).
    """
    def __init__(self, num_workers, eval_function):
        """
        eval_function should take two arguments (a genome object and the
        configuration) and return a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.workers = []
        self.working = False
        self.inqueue = queue.Queue()
        self.outqueue = queue.Queue()

        if not HAVE_THREADS: # pragma: no cover
            warnings.warn("No threads available; use ParallelEvaluator, not ThreadedEvaluator")

    def __del__(self):
        """
        Called on deletion of the object. We stop our workers here.
        WARNING: __del__ may not always work!
        Please stop the threads explicitly by calling self.stop()!
        TODO: ensure that there are no reference-cycles.
        """
        if self.working:
            self.stop()

    def start(self):
        """Starts the worker threads"""
        if self.working:
            return
        self.working = True
        for i in range(self.num_workers):
            w = threading.Thread(
                name="Worker Thread #{i}".format(i=i),
                target=self._worker,
                )
            w.daemon = True
            w.start()
            self.workers.append(w)

    def stop(self):
        """Stops the worker threads and waits for them to finish"""
        self.working = False
        for w in self.workers:
            w.join()
        self.workers = []

    def _worker(self):
        """The worker function"""
        while self.working:
            try:
                genome_id, genome, config = self.inqueue.get(
                    block=True,
                    timeout=0.2,
                    )
            except queue.Empty:
                continue
            f = self.eval_function(genome, config)
            self.outqueue.put((genome_id, genome, f))

    def evaluate(self, genomes, config):
        """Evaluate the genomes"""
        if not self.working:
            self.start()
        p = 0
        for genome_id, genome in genomes:
            p += 1
            self.inqueue.put((genome_id, genome, config))

        # assign the fitness back to each genome
        while p > 0:
            p -= 1
            ignored_genome_id, genome, fitness = self.outqueue.get()
            genome.fitness = fitness
