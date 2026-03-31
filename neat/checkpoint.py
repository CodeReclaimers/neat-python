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

    Checkpoints are saved after fitness evaluation (in ``post_evaluate``), so the
    saved population contains genomes with their evaluated fitness values.  This
    means restoring a checkpoint never re-evaluates work that was already done.

    The checkpoint filename suffix (for example, ``neat-checkpoint-10``) refers to
    the generation that has just been **evaluated**.  Restoring checkpoint ``N``
    reproduces from generation ``N``'s evaluated results and then continues
    evaluation from generation ``N + 1``.
    """

    def __init__(self, generation_interval, time_interval_seconds=None,
                 filename_prefix='neat-checkpoint-'):
        """
        Saves the current state (after fitness evaluation) every
        ``generation_interval`` generations or ``time_interval_seconds``,
        whichever happens first.

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

    def post_evaluate(self, config, population, species, best_genome):
        """Potentially save a checkpoint after fitness evaluation.

        At this point the population has been evaluated and species membership
        corresponds to the evaluated genomes, so the checkpoint captures a
        fully consistent state with no wasted work on restore.
        """
        checkpoint_due = False

        if self.time_interval_seconds is not None:
            dt = time.time() - self.last_time_checkpoint
            if dt >= self.time_interval_seconds:
                checkpoint_due = True

        if (not checkpoint_due) and (self.generation_interval is not None):
            dg = self.current_generation - self.last_generation_checkpoint
            if dg >= self.generation_interval:
                checkpoint_due = True

        if checkpoint_due:
            self.save_checkpoint(config, population, species,
                                 self.current_generation, best_genome)
            self.last_generation_checkpoint = self.current_generation
            self.last_time_checkpoint = time.time()

    def save_checkpoint(self, config, population, species_set, generation, best_genome=None):
        """
        Save the current simulation state.

        The saved data includes the evaluated population (with fitness values),
        the species set, the generation index, the all-time best genome, and the
        random state for reproducibility.
        """
        filename = f'{self.filename_prefix}{generation}'
        print(f"Saving checkpoint to {filename}")

        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (generation, config, population, species_set,
                    random.getstate(), best_genome)
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_checkpoint(filename, new_config=None):
        """
        Resumes the simulation from a previous saved point.

        The checkpoint contains the evaluated population from generation ``N``.
        On restore, evaluation is skipped for this generation and the evolution
        loop proceeds directly to reproduction, continuing with generation
        ``N + 1``.

        The innovation tracker state is preserved in the pickled config and
        transferred to the new reproduction object to ensure innovation numbers
        continue correctly.
        """
        with gzip.open(filename) as f:
            data = pickle.load(f)
            # Support both old (5-tuple) and new (6-tuple) checkpoint formats.
            if len(data) == 6:
                generation, saved_config, population, species_set, rndstate, best_genome = data
            else:
                generation, saved_config, population, species_set, rndstate = data
                best_genome = None

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
            restored_pop = Population(config, (population, species_set, generation))

            # Restore best_genome so the all-time best is not lost
            if best_genome is not None:
                restored_pop.best_genome = best_genome

            # Tell run() to skip the first evaluation — it was already done
            # before this checkpoint was saved.
            restored_pop._skip_first_evaluation = True

            # Replace the fresh innovation tracker with the saved one to maintain
            # the correct innovation numbering sequence
            if saved_innovation_tracker is not None:
                restored_pop.reproduction.innovation_tracker = saved_innovation_tracker
                config.genome_config.innovation_tracker = saved_innovation_tracker

            return restored_pop
