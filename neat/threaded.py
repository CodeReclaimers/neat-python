"""threaded evaluation of genomes"""
import threading

try:
    import Queue as queue
except ImportError:
    import queue


class ThreadedEvaluator(object):
    """
    A threaded genome evaluator.
    Usefull on python implementations without GIL.
    """
    def __init__(self, num_workers, eval_function):
        '''
        eval_function should take one argument (a genome object) and return
        a single float (the genome's fitness).
        '''
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.workers = []
        self.working = False
        self.inqueue = queue.Queue()
        self.outqueue = queue.Queue()

    def __del__(self):
        if self.working:
            self.stop()
            
    def start(self):
        """starts the worker threads"""
        if self.working:
            return
        self.working = True
        for i in xrange(self.num_workers):
            w = threading.Thread(
                name="Worker Thread #{i}".format(i=i),
                target=self._worker,
                )
            w.daemon = True
            w.start()
            self.workers.append(w)

    def stop(self):
        """stops the worker threads and wait for them to finish"""
        self.working = False
        for w in self.workers:
            w.join()

    def _worker(self):
        """the worker function"""
        while self.working:
            try:
                genome_id, genome = self.inqueue.get(block=True, timeout=0.2)
            except queue.Empty:
                continue
            f = self.eval_function(genome)
            self.outqueue.put((genome_id, genome, f))

    def evaluate(self, genomes, config):
        """evaluate the genomes"""
        p = 0
        for genome_id, genome in genomes:
            p += 1
            self.inqueue.put((genome_id, genome))

        # assign the fitness back to each genome
        while p > 0:
            p -= 1
            genome_id, genome, fitness = self.outqueue.get()
            genome.fitness = fitness