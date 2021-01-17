"""Divides the population into species based on genomic distances."""
from __future__ import annotations

from itertools import count
from typing import List, Dict, Set, Tuple, Optional, TYPE_CHECKING

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean, stdev

if TYPE_CHECKING:
    from neat.config import DefaultClassConfig, Config
    from neat.reporting import ReporterSet
    from neat.genome import DefaultGenome, DefaultGenomeConfig


class Species(object):
    def __init__(self, key: int, generation: int):
        self.key: int = key
        self.created: int = generation
        self.last_improved: int = generation
        self.representative: Optional[DefaultGenome] = None
        self.members: Dict[int, DefaultGenome] = {}
        self.fitness: Optional[float] = None
        self.adjusted_fitness: Optional[float] = None
        self.fitness_history: List[float] = []

    def update(self, representative: DefaultGenome, members: Dict[int, DefaultGenome]):
        self.representative = representative
        self.members = members

    def get_fitnesses(self) -> List[float]:
        return [m.fitness for m in self.members.values()]


class GenomeDistanceCache(object):
    def __init__(self, config: DefaultGenomeConfig):
        self.distances: Dict[Tuple[int, int], float] = {}
        self.config: DefaultGenomeConfig = config
        self.hits: int = 0
        self.misses: int = 0

    def __call__(self, genome0, genome1) -> float:
        g0: int = genome0.key
        g1: int = genome1.key
        d: float = self.distances.get((g0, g1))
        if d is None:
            # Distance is not already computed.
            d = genome0.distance(genome1, self.config)
            self.distances[g0, g1] = d
            self.distances[g1, g0] = d
            self.misses += 1
        else:
            self.hits += 1

        return d


class DefaultSpeciesSet(DefaultClassConfig):
    """ Encapsulates the default speciation scheme. """

    def __init__(self, config: DefaultClassConfig, reporters: ReporterSet):
        # pylint: disable=super-init-not-called
        self.species_set_config: DefaultClassConfig = config
        self.reporters: ReporterSet = reporters
        self.indexer: count = count(1)
        self.species: Dict[int, Species] = {}
        self.genome_to_species: Dict[int, int] = {}

    @classmethod
    def parse_config(cls, param_dict: Dict[str, str]) -> DefaultClassConfig:
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float)])

    def speciate(self, config: Config, population: Dict[int, DefaultGenome], generation: int):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert isinstance(population, dict)

        compatibility_threshold: float = self.species_set_config.compatibility_threshold

        # Find the best representatives for each existing species.
        unspeciated: Set[int] = set(population)
        distances: GenomeDistanceCache = GenomeDistanceCache(config.genome_config)
        new_representatives: Dict[int, int] = {}
        new_members: Dict[int, List[int]] = {}
        for sid, s in self.species.items():
            candidates: List[Tuple[float, DefaultGenome]] = []
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid: int = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)

        # Partition population into species based on genetic similarity.
        while unspeciated:
            gid: int = unspeciated.pop()
            g: DefaultGenome = population[gid]

            # Find the species with the most similar representative.
            candidates: List[Tuple[float, int]] = []
            for sid, rid in new_representatives.items():
                rep: DefaultGenome = population[rid]
                d: float = distances(rep, g)
                if d < compatibility_threshold:
                    candidates.append((d, sid))

            if candidates:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid: int = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        self.genome_to_species: Dict[int, int] = {}
        for sid, rid in new_representatives.items():
            s: Optional[Species] = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members: List[int] = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict: Dict[int, DefaultGenome] = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        gdmean: float = mean(distances.distances.values())
        gdstdev: float = stdev(distances.distances.values())
        self.reporters.info(
            'Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev))

    def get_species_id(self, individual_id: int) -> int:
        """
        個体iが所属する種のIDを返す
        """
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id: int):
        """
        個体iが所属する種を返す
        """
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
