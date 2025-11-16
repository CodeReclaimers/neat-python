"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
import random
from multiprocessing import Pool

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False
    # Fallback: tqdm is just an identity function when not available
    def tqdm(iterable, total=None):
        return iterable


def _eval_wrapper(eval_function, seed, genome, config):
    """
    Wrapper that sets deterministic per-genome seed before evaluation.
    
    Each genome gets a unique but deterministic seed based on:
    - Base seed provided to ParallelEvaluator
    - Genome's unique key (ID)
    
    This ensures:
    - Different genomes can use randomness in fitness evaluation
    - Same genome always gets same random sequence (given same base seed)
    - Results are reproducible across runs
    
    Note: Only controls Python's random module. If fitness function uses
    NumPy, PyTorch, etc., those RNGs need separate seeding.
    
    Args:
        eval_function: The user-provided fitness evaluation function
        seed: Base seed for reproducibility (or None for non-deterministic)
        genome: Genome object to evaluate
        config: Configuration object
    
    Returns:
        Fitness value for the genome
    """
    if seed is not None:
        # Use genome key to ensure unique but deterministic seed per genome
        random.seed(seed + genome.key)
    return eval_function(genome, config)

class ParallelEvaluator:
    def __init__(self, num_workers, eval_function, timeout=None, initializer=None, initargs=(), maxtasksperchild=None, seed=None):
        """
        Parallel fitness evaluator using multiprocessing.
        
        Args:
            num_workers: Number of worker processes to use
            eval_function: Function that takes (genome, config) and returns fitness
            timeout: Optional timeout for fitness evaluation
            initializer: Optional function to initialize worker processes
            initargs: Arguments for initializer function
            maxtasksperchild: Maximum tasks per worker before restart
            seed: Optional random seed for reproducible parallel evaluation.
                  If provided, each genome will receive a deterministic seed
                  based on: seed + genome.key
                  If None (default), behavior is non-deterministic.
        
        Note on reproducibility:
            Setting a seed ensures that the same genome evaluated with the same
            base seed will always produce the same fitness value (assuming the
            fitness function uses Python's random module). Each genome gets a
            unique but deterministic seed (base_seed + genome_id) so different
            genomes get different random sequences.
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.seed = seed
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
            if self.seed is not None:
                # Use wrapper to set per-genome seed for reproducibility
                jobs.append(self.pool.apply_async(
                    _eval_wrapper,
                    (self.eval_function, self.seed, genome, config)
                ))
            else:
                # Original behavior - no seed
                jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in tqdm(zip(jobs, genomes), total=len(jobs)):
            genome.fitness = job.get(timeout=self.timeout)
