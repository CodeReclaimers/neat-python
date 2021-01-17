"""
Gathers (via the reporting interface) and provides (to callers and/or a file)
the most-fit genomes and information on genome/species fitness and species sizes.
"""
from __future__ import annotations
import copy
import csv

from typing import List, Set, Tuple, Union, Optional, Callable, Dict, TYPE_CHECKING
from neat.math_util import mean, stdev, median2
from neat.reporting import BaseReporter

if TYPE_CHECKING:
    from neat.genome import DefaultGenome
    from neat.config import Config
    from neat.species import DefaultSpeciesSet


# TODO: Make a version of this reporter that doesn't continually increase memory usage.
# (Maybe periodically write blocks of history to disk, or log stats in a database?)

class StatisticsReporter(BaseReporter):
    """
    Gathers (via the reporting interface) and provides (to callers and/or a file)
    the most-fit genomes and information on genome/species fitness and species sizes.
    """

    def __init__(self):
        BaseReporter.__init__(self)
        self.most_fit_genomes: List[DefaultGenome] = []
        self.generation_statistics: List[Dict[int, Dict[int, int]]] = []

    def post_evaluate(self, config: Config, population: Dict[int, DefaultGenome], species_set: DefaultSpeciesSet, best_genome: DefaultGenome) -> None:
        self.most_fit_genomes.append(copy.deepcopy(best_genome))

        # Store the fitnesses of the members of each currently active species.
        species_stats: Dict[int, Dict[int, int]] = {}
        for sid, s in species_set.species.items():
            species_stats[sid] = dict((k, v.fitness) for k, v in s.members.items())
        self.generation_statistics.append(species_stats)

    def get_fitness_stat(self, f: Callable) -> List[float]:
        """
        種ごとに関数fを書けた適応度を返す　４種なら４のリスト
        """
        stat: List[float] = []
        for stats in self.generation_statistics:
            scores: List[float] = []
            for species_stats in stats.values():
                scores.extend(species_stats.values())
            stat.append(f(scores))

        return stat

    def get_fitness_mean(self) -> List[float]:
        """Get the per-generation mean fitness."""
        return self.get_fitness_stat(mean)

    def get_fitness_stdev(self) -> List[float]:
        """Get the per-generation standard deviation of the fitness."""
        return self.get_fitness_stat(stdev)

    def get_fitness_median(self) -> List[float]:
        """Get the per-generation median fitness."""
        return self.get_fitness_stat(median2)

    def best_unique_genomes(self, n: int) -> List[DefaultGenome]:
        """Returns the most n fit genomes, with no duplication."""
        best_unique: Dict[int, DefaultGenome] = {}
        for g in self.most_fit_genomes:
            best_unique[g.key] = g
        best_unique_list: List[DefaultGenome] = list(best_unique.values())

        def key(genome):
            return genome.fitness

        return sorted(best_unique_list, key=key, reverse=True)[:n]

    def best_genomes(self, n: int) -> List[DefaultGenome]:
        """Returns the n most fit genomes ever seen."""

        def key(g):
            return g.fitness

        return sorted(self.most_fit_genomes, key=key, reverse=True)[:n]

    def best_genome(self) -> DefaultGenome:
        """Returns the most fit genome ever seen."""
        return self.best_genomes(1)[0]

    def save(self) -> None:
        self.save_genome_fitness()
        self.save_species_count()
        self.save_species_fitness()

    def save_genome_fitness(self,
                            delimiter: str = ' ',
                            filename: str = 'fitness_history.csv'):
        """ Saves the population's best and average fitness. """
        with open(filename, 'w') as f:
            w = csv.writer(f, delimiter=delimiter)

            # 世代の長さ　各世代の適応度
            best_fitness: List[float] = [c.fitness for c in self.most_fit_genomes]
            avg_fitness: List[float] = self.get_fitness_mean()

            for best, avg in zip(best_fitness, avg_fitness):
                w.writerow([best, avg])

    def save_species_count(self, delimiter: str = ' ', filename: str = 'speciation.csv'):
        """ Log speciation throughout evolution. """
        with open(filename, 'w') as f:
            w = csv.writer(f, delimiter=delimiter)
            for s in self.get_species_sizes():
                w.writerow(s)

    def save_species_fitness(self, delimiter=' ', null_value='NA', filename='species_fitness.csv'):
        """ Log species' average fitness throughout evolution. """
        with open(filename, 'w') as f:
            w = csv.writer(f, delimiter=delimiter)
            for s in self.get_species_fitness(null_value):
                w.writerow(s)

    def get_species_sizes(self) -> List[List[int]]:

        # おそらくこれまでに出てきた種のIDのセット
        all_species: Set[int] = set()
        for gen_data in self.generation_statistics:
            all_species: Set[int] = all_species.union(gen_data.keys())

        max_species: int = max(all_species)

        # 世代ごとの各種の個体数
        # species_counts[20][2] = 20世代目における種２の個体数
        species_counts: List[List[int]] = []
        for gen_data in self.generation_statistics:
            species = [len(gen_data.get(sid, [])) for sid in range(1, max_species + 1)]
            species_counts.append(species)

        return species_counts

    def get_species_fitness(self, null_value: str = '') -> List[List[Union[str, float]]]:
        all_species: Set[int] = set()
        for gen_data in self.generation_statistics:
            all_species: Set[int] = all_species.union(gen_data.keys())

        max_species: int = max(all_species)

        # 各世代における各種の適応度の平均
        # ある世代で一体も種がないときはNAが入る
        species_fitness: List[List[Union[str, float]]] = []
        for gen_data in self.generation_statistics:

            # ある世代における　各種に含まれる各個体の適応度
            member_fitness: List[Union[Dict[int, float], List]] = [gen_data.get(sid, []) for sid in range(1, max_species + 1)]
            fitness: List[Union[str, float]] = []
            for mf in member_fitness:
                if mf:
                    fitness.append(mean(mf))
                else:
                    fitness.append(null_value)
            species_fitness.append(fitness)

        return species_fitness
