"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False
    # Fallback: tqdm is just an identity function when not available
    def tqdm(iterable, total=None):
        return iterable

class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None, initializer=None, initargs=(), maxtasksperchild=None):
        """
        eval_function should take one argument, a tuple of (genome object, config object),
        and return a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.initializer = initializer
        self.initargs = initargs
        self.maxtasksperchild = maxtasksperchild
        self.pool = Pool(processes=num_workers, maxtasksperchild=maxtasksperchild, initializer=initializer, initargs=initargs)
        self._closed = False

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point - ensures proper cleanup."""
        self.close()
        return False

    def close(self):
        """Properly close and cleanup the multiprocessing pool."""
        if self.pool is not None and not self._closed:
            self._closed = True
            self.pool.close()  # Prevent any more tasks from being submitted
            self.pool.join()   # Wait for worker processes to exit
            self.pool = None

    def __del__(self):
        """Cleanup on deletion - ensures resources are freed."""
        self.close()

    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in tqdm(zip(jobs, genomes), total=len(jobs)):
            genome.fitness = job.get(timeout=self.timeout)