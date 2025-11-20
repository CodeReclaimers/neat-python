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

        The checkpoint filename suffix (for example, ``neat-checkpoint-10``) always refers to the
        **next generation to be evaluated**.  In other words, a checkpoint created with suffix ``N``
        contains the population and species state for generation ``N`` at the point just before
        its fitness evaluation begins.

        :param generation_interval: If not None, maximum number of generations between save intervals,
                                    measured in generations-to-be-evaluated
        :type generation_interval: int or None
        :param time_interval_seconds: If not None, maximum number of seconds between checkpoint attempts
        :type time_interval_seconds: float or None
        :param str filename_prefix: Prefix for the filename (the end will be the generation number)
        """
        self.generation_interval = generation_interval
        self.time_interval_seconds = time_interval_seconds
        self.filename_prefix = filename_prefix

        self.current_generation = None
        # Tracks the most recent generation index for which a checkpoint was created.
        # This value is interpreted as the next generation to be evaluated when the
        # checkpoint is restored (see above).
        self.last_generation_checkpoint = 0
        self.last_time_checkpoint = time.time()

    def start_generation(self, generation):
        """Record the index of the generation that is about to be evaluated.

        Note that at the time :meth:`end_generation` is called for generation ``g``,
        the population and species that are passed in already correspond to the
        *next* generation (``g + 1``).  This reporter therefore uses ``g + 1`` as
        the generation index stored in checkpoints, so that restoring a
        checkpoint labeled ``N`` always resumes at the beginning of generation
        ``N``.
        """
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        """Potentially save a checkpoint at the end of a generation.

        The ``population`` and ``species_set`` arguments contain the state for
        the next generation to be evaluated, whose index is
        ``self.current_generation + 1``.
        """
        checkpoint_due = False

        if self.time_interval_seconds is not None:
            dt = time.time() - self.last_time_checkpoint
            if dt >= self.time_interval_seconds:
                checkpoint_due = True

        # The generation whose population is being saved.
        next_generation = self.current_generation + 1

        if (not checkpoint_due) and (self.generation_interval is not None):
            # Compare the upcoming generation index against the last checkpointed
            # generation index to decide whether a new checkpoint is due.
            dg = next_generation - self.last_generation_checkpoint
            if dg >= self.generation_interval:
                checkpoint_due = True

        if checkpoint_due:
            self.save_checkpoint(config, population, species_set, next_generation)
            self.last_generation_checkpoint = next_generation
            self.last_time_checkpoint = time.time()

    def save_checkpoint(self, config, population, species_set, generation):
        """
        Save the current simulation state.
        
        Note: This is called from Population via the reporter interface.
        We need to access the innovation tracker from the Population's reproduction object.
        However, since this is a reporter callback, we don't have direct access to Population.
        The innovation tracker will be saved as part of the config state when needed.
        """
        filename = f'{self.filename_prefix}{generation}'
        print(f"Saving checkpoint to {filename}")

        with gzip.open(filename, 'w', compresslevel=5) as f:
            # Note: innovation_tracker is stored in config.genome_config.innovation_tracker
            # and is automatically included via pickle
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_checkpoint(filename, new_config=None):
        """
        Resumes the simulation from a previous saved point.
        
        The innovation tracker state is preserved in the pickled config and must be
        transferred to the new reproduction object to ensure innovation numbers continue
        correctly and prevent collisions during crossover.
        """
        with gzip.open(filename) as f:
            generation, saved_config, population, species_set, rndstate = pickle.load(f)
            random.setstate(rndstate)
            
            # Extract the saved innovation tracker from the config before replacing it
            saved_innovation_tracker = None
            if hasattr(saved_config.genome_config, 'innovation_tracker'):
                saved_innovation_tracker = saved_config.genome_config.innovation_tracker
            
            # Use new config if provided, otherwise use saved config
            if new_config is not None:
                config = new_config
            else:
                config = saved_config
            
            # Create Population with restored state
            # This creates a new reproduction object with a fresh innovation tracker
            restored_pop = Population(config, (population, species_set, generation))
            
            # Replace the fresh innovation tracker with the saved one to maintain
            # the correct innovation numbering sequence
            if saved_innovation_tracker is not None:
                restored_pop.reproduction.innovation_tracker = saved_innovation_tracker
                config.genome_config.innovation_tracker = saved_innovation_tracker
            
            return restored_pop
