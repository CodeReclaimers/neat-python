"""
Implementation of reporter classes, which are triggered on particular events. Reporters
are generally intended to  provide information to the user, store checkpoints, etc.
"""

import time

from neat.math_util import mean, stdev


class ReporterSet:
    """
    Keeps track of the set of reporters
    and gives methods to dispatch them at appropriate points.
    """

    def __init__(self):
        self.reporters = []

    def add(self, reporter):
        self.reporters.append(reporter)

    def remove(self, reporter):
        self.reporters.remove(reporter)

    def start_generation(self, gen):
        for r in self.reporters:
            r.start_generation(gen)

    def end_generation(self, config, population, species_set):
        for r in self.reporters:
            r.end_generation(config, population, species_set)

    def post_evaluate(self, config, population, species, best_genome):
        for r in self.reporters:
            r.post_evaluate(config, population, species, best_genome)

    def post_reproduction(self, config, population, species):
        for r in self.reporters:
            r.post_reproduction(config, population, species)

    def complete_extinction(self):
        for r in self.reporters:
            r.complete_extinction()

    def found_solution(self, config, generation, best):
        for r in self.reporters:
            r.found_solution(config, generation, best)

    def species_stagnant(self, sid, species):
        for r in self.reporters:
            r.species_stagnant(sid, species)

    def info(self, msg):
        for r in self.reporters:
            r.info(msg)


class BaseReporter:
    """Definition of the reporter interface expected by ReporterSet."""

    def start_generation(self, generation):
        pass

    def end_generation(self, config, population, species_set):
        pass

    def post_evaluate(self, config, population, species, best_genome):
        pass

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass


class StdOutReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""

    def __init__(self, show_species_detail):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

    def start_generation(self, generation):
        self.generation = generation
        print(f'\n ****** Running generation {generation} ****** \n')
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            print(f'Population of {ng:d} members in {ns:d} species (after reproduction):')
            print("   ID   age  size   fitness   adj fit  stag")
            print("  ====  ===  ====  =========  =======  ====")
            for sid in sorted(species_set.species):
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else f"{s.fitness:.3f}"
                af = "--" if s.adjusted_fitness is None else f"{s.adjusted_fitness:.3f}"
                st = self.generation - s.last_improved
                print(f"  {sid:>4}  {a:>3}  {n:>4}  {f:>9}  {af:>7}  {st:>4}")
        else:
            print(f'Population of {ng:d} members in {ns:d} species (after reproduction)')

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print(f'Total extinctions: {self.num_extinctions:d}')
        if len(self.generation_times) > 1:
            print(f"Generation time: {elapsed:.3f} sec ({average:.3f} average)")
        else:
            print(f"Generation time: {elapsed:.3f} sec")

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print(f'Population\'s average fitness: {fit_mean:3.5f} stdev: {fit_std:3.5f}')
        print(
            'Best fitness: {:3.5f} - size: {!r} - species {} - id {}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self, config, generation, best):
        print('\nBest individual in generation {} meets fitness threshold - complexity: {!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            print(f"\nSpecies {sid} with {len(species.members)} members is stagnated: removing it")

    def info(self, msg):
        print(msg)
