from multiprocessing import Pool


class ParallelEvaluator:
    def __init__(self, num_workers, eval_function, timeout=1):
        '''
        eval_function should take one argument (a genome object) and return
        a single float (the genome's fitness).
        '''
        self.num_workers = 6
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)

    def evaluate(self, genomes):
        jobs = []
        for genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome,)))

        # assign the fitness back to each genome
        for job, genome in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
