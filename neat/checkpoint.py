"""Uses `pickle` to save and restore populations (and other aspects of the simulation state)."""

import gzip
import pickle
import random
import time

from neat.population import Population
from neat.reporting import BaseReporter


class Checkpointer(BaseReporter):
    """
    A reporter class that performs checkpointing using `pickle`
    to save and restore populations (and other aspects of the simulation state).
    """

    def __init__(self, generation_interval, time_interval_seconds=None,
                 filename_prefix='neat-checkpoint-'):
        """
        Saves the current state (at the end of a generation) every ``generation_interval`` generations or
        ``time_interval_seconds``, whichever happens first.

        :param generation_interval: If not None, maximum number of generations between save intervals
        :type generation_interval: int or None
        :param time_interval_seconds: If not None, maximum number of seconds between checkpoint attempts
        :type time_interval_seconds: float or None
        :param str filename_prefix: Prefix for the filename (the end will be the generation number)
        """
        self.generation_interval = generation_interval
        self.time_interval_seconds = time_interval_seconds
        self.filename_prefix = filename_prefix

        self.current_generation = None
        self.last_generation_checkpoint = -1
        self.last_time_checkpoint = time.time()

    def start_generation(self, generation):
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        checkpoint_due = False

        if self.time_interval_seconds is not None:
            dt = time.time() - self.last_time_checkpoint
            if dt >= self.time_interval_seconds:
                checkpoint_due = True

        if (checkpoint_due is False) and (self.generation_interval is not None):
            dg = self.current_generation - self.last_generation_checkpoint
            if dg >= self.generation_interval:
                checkpoint_due = True

        if checkpoint_due:
            self.save_checkpoint(config, population, species_set, self.current_generation)
            self.last_generation_checkpoint = self.current_generation
            self.last_time_checkpoint = time.time()

    def save_checkpoint(self, config, population, species_set, generation):
        """
        Save the current simulation state.
        
        Note: This is called from Population via the reporter interface.
        We need to access the innovation tracker from the Population's reproduction object.
        However, since this is a reporter callback, we don't have direct access to Population.
        The innovation tracker will be saved as part of the config state when needed.
        """
        filename = '{0}{1}'.format(self.filename_prefix, generation)
        print("Saving checkpoint to {0}".format(filename))

        with gzip.open(filename, 'w', compresslevel=5) as f:
            # Note: innovation_tracker is stored in config.genome_config.innovation_tracker
            # and is automatically included via pickle
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_checkpoint(filename, new_config=None):
        """
        Resumes the simulation from a previous saved point.
        
        The innovation tracker state is preserved through the Population's reproduction object,
        which is automatically pickled and restored. The global innovation counter will continue
        from where it left off, preventing innovation number collisions.
        """
        with gzip.open(filename) as f:
            generation, config, population, species_set, rndstate = pickle.load(f)
            random.setstate(rndstate)
            if new_config is not None:
                config = new_config
            
            # Create Population with restored state
            # The Population.__init__ will restore the reproduction object which contains
            # the innovation_tracker with its preserved state
            restored_pop = Population(config, (population, species_set, generation))
            
            # The innovation tracker should already be restored via pickle, but we need to
            # ensure it's properly connected to the genome_config for the next generation
            if hasattr(restored_pop.reproduction, 'innovation_tracker'):
                config.genome_config.innovation_tracker = restored_pop.reproduction.innovation_tracker
            
            return restored_pop
