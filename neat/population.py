from __future__ import print_function

from neat.reporting import ReporterSet
from neat.six_util import iteritems, itervalues


class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """ This class implements the core NEAT algorithm. """

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config, self.reporters, stagnation)

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type, config.genome_config, config.pop_size)
            self.species = config.species_set_type(config, self.reporters)
            self.species.speciate(config, self.population)
            self.generation = -1
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def run(self, fitness_function, n):
        """
        Runs NEAT's genetic algorithm for at most n generations.

        The user-provided fitness_function should take two arguments, a list of all genomes in the population,
        and its return value is ignored.  This function is free to maintain external state, perform evaluations
        in parallel, and probably any other thing you want.  Each individual genome's fitness member must be set
        to a floating point value after this function returns.

        It is assumed that fitness_function does not modify the list of genomes, the genomes themselves (apart
        from updating the fitness member), or the configuration object.
        """
        for g in range(n):
            self.generation += 1

            self.reporters.start_generation(self.generation)

            # Evaluate all individuals in the population using the user-provided function.
            # TODO: Add an option to only evaluate each genome once, to reduce number of
            # fitness evaluations in cases where the fitness is known to be the same if the
            # genome doesn't change--in these cases, evaluating unmodified elites in each
            # generation is a waste of time.  The user can always take care of this in their
            # fitness function in the time being if they wish.
            fitness_function(list(iteritems(self.population)), self.config)

            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            # End if the fitness threshold is reached.
            if best.fitness >= self.config.max_fitness_threshold:
                self.reporters.found_solution(self.config, self.generation, best)
                break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species, self.config.pop_size)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Update species age.
            # TODO: Wouldn't it be easier to remember creation time?
            for s in itervalues(self.species.species):
                s.age += 1

            # Divide the new population into species.
            self.species.speciate(self.config, self.population)

            self.reporters.end_generation(self.config, self.population, self.species)

        return self.best_genome