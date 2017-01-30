from __future__ import print_function
import gzip
import pickle
import random
import time

import neat
from neat.reporting import BaseReporter


class Checkpointer(BaseReporter):
    def __init__(self, generation_interval=100, time_interval_seconds=300):
        self.generation_interval = generation_interval
        self.time_interval_seconds = time_interval_seconds

        self.current_generation = None
        self.last_generation_checkpoint = -1
        self.last_time_checkpoint = time.time()

    def start_generation(self, generation):
        self.current_generation = generation

    def end_generation(self, config, population, species):
        checkpoint_due = False

        if self.time_interval_seconds is not None:
            dt = time.time() - self.last_time_checkpoint
            if dt >= self.time_interval_seconds:
                checkpoint_due = True

        if self.generation_interval is not None:
            dg = self.current_generation - self.last_generation_checkpoint
            if dg >= self.generation_interval:
                checkpoint_due = True

        if checkpoint_due:
            self.save_checkpoint(config, population, species, self.current_generation)
            self.last_generation_checkpoint = self.current_generation
            self.last_time_checkpoint = time.time()

    @staticmethod
    def save_checkpoint(config, population, species, generation):
        """ Save the current simulation state. """
        filename = 'neat-checkpoint-{0}'.format(generation)
        print("Saving checkpoint to {0}".format(filename))

        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (generation, config, population, species, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_checkpoint(filename):
        '''Resumes the simulation from a previous saved point.'''
        with gzip.open(filename) as f:
            generation, config, population, species, rndstate = pickle.load(f)
            random.setstate(rndstate)
            return neat.Population(config, (population, species, generation))


