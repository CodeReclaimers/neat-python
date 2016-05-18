from __future__ import print_function

import copy
import time

from neat.math_util import mean, stdev


class ReporterSet(object):
    def __init__(self):
        self.reporters = []

    def add(self, reporter):
        self.reporters.append(reporter)

    def remove(self, reporter):
        self.reporters.remove(reporter)

    def start_generation(self, gen):
        for r in self.reporters:
            r.start_generation(gen)

    def end_generation(self):
        for r in self.reporters:
            r.end_generation()

    def loading_checkpoint(self, filename):
        for r in self.reporters:
            r.loading_checkpoint(filename)

    def saving_checkpoint(self, checkpoint_type, filename):
        for r in self.reporters:
            r.saving_checkpoint(checkpoint_type, filename)

    def post_evaluate(self, population, species, best):
        for r in self.reporters:
            r.post_evaluate(population, species, best)

    def complete_extinction(self):
        for r in self.reporters:
            r.complete_extinction()

    def found_solution(self, generation, best):
        for r in self.reporters:
            r.found_solution(generation, best)

    def species_stagnant(self, species):
        for r in self.reporters:
            r.species_stagnant(species)

    def info(self, msg):
        for r in self.reporters:
            r.info(msg)


class BaseReporter(object):
    """Definition of the reporter interface expected by ReporterSet."""
    def start_generation(self, generation):
        pass

    def end_generation(self):
        pass

    def loading_checkpoint(self, filename):
        pass

    def saving_checkpoint(self, checkpoint_type, filename):
        pass

    def post_evaluate(self, population, species, best):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, generation, best):
        pass

    def species_stagnant(self, species):
        pass

    def info(self, msg):
        pass


class StdOutReporter(BaseReporter):
    def __init__(self):
        self.generation = None
        self.generation_start_time = None

    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self):
        print("Generation time: {0:.3f} sec".format(time.time() - self.generation_start_time))

    def loading_checkpoint(self, filename):
        print('Resuming from a previous point: ' + filename)

    def saving_checkpoint(self, checkpoint_type, filename):
        print('Creating {0} checkpoint file {1} at generation: {0}'.format(
            checkpoint_type, filename, self.generation))

    def post_evaluate(self, population, species, best):
        fit_mean = mean([c.fitness for c in population])
        fit_std = stdev([c.fitness for c in population])
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print('Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best.fitness, best.size(),
                                                                                   best.species_id, best.ID))
        print('Species length: {0:d} totaling {1:d} individuals'.format(len(species), sum(
            [len(s.members) for s in species])))
        print('Species ID       : {0!s}'.format([s.ID for s in species]))
        print('Species size     : {0!s}'.format([len(s.members) for s in species]))
        print('Species age      : {0}'.format([s.age for s in species]))

    def complete_extinction(self):
        print('All species extinct.')

    def found_solution(self, generation, best):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, species):
        print("\nSpecies {0} with {1} members is stagnated: removing it".format(species.ID, len(species.members)))

    def info(self, msg):
        print(msg)


class StatisticsReporter(BaseReporter):
    def __init__(self):
        BaseReporter.__init__(self)
        self.most_fit_genomes = []
        self.generation_statistics = []

    def post_evaluate(self, population, species, best):
        self.most_fit_genomes.append(copy.deepcopy(best))

        # Store the fitnesses of the members of each currently active species.
        species_stats = {}
        for s in species:
            species_stats[s.ID] = [m.fitness for m in s.members]
        self.generation_statistics.append(species_stats)

    def get_average_fitness(self):
        """Get the per-generation average fitness."""
        avg_fitness = []
        for stats in self.generation_statistics:
            scores = []
            for fitness in stats.values():
                scores.extend(fitness)
            avg_fitness.append(mean(scores))

        return avg_fitness

    def best_unique_genomes(self, n):
        """Returns the most n fit genomes, with no duplication."""
        best_unique = {}
        for g in self.most_fit_genomes:
            best_unique[g.ID] = g
        best_unique = list(best_unique.values())

        def key(genome):
            return genome.fitness

        return sorted(best_unique, key=key, reverse=True)[:n]

    def best_genomes(self, n):
        """Returns the n most fit genomes ever seen."""
        def key(g):
            return g.fitness

        return sorted(self.most_fit_genomes, key=key, reverse=True)[:n]

    def best_genome(self):
        """Returns the most fit genome ever seen."""
        return self.best_genomes(1)[0]

