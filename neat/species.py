"""Divides the population into species based on genomic distances."""
from itertools import count

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean, stdev


class Species:
    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative = None
        self.members = {}
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []

    def update(self, representative, members):
        self.representative = representative
        self.members = members

    def get_fitnesses(self):
        return [m.fitness for m in self.members.values()]


class GenomeDistanceCache:
    def __init__(self, config):
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome0, genome1):
        g0 = genome0.key
        g1 = genome1.key
        d = self.distances.get((g0, g1))
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

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float)],
                                  'DefaultSpeciesSet')

    def speciate(self, config, population, generation):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert isinstance(population, dict)

        compatibility_threshold = self.species_set_config.compatibility_threshold

        # Find the best representatives for each existing species.
        # Use a deterministic ordering for unspeciated genomes so that
        # speciation is reproducible across runs and checkpoint restores.
        unspeciated = list(sorted(population.keys()))
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}
        new_members = {}
        # Iterate species in deterministic id order.
        for sid in sorted(self.species.keys()):
            s = self.species[sid]
            candidates = []
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)

        # Partition population into species based on genetic similarity.
        # Iterate remaining genomes in ascending id order for determinism.
        while unspeciated:
            gid = unspeciated.pop(0)
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in new_representatives.items():
                rep = population[rid]
                d = distances(rep, g)
                if d < compatibility_threshold:
                    candidates.append((d, sid))

            if candidates:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid in sorted(new_representatives.keys()):
            rid = new_representatives[sid]
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = {gid: population[gid] for gid in members}
            s.update(population[rid], member_dict)

        # Mean and std genetic distance info report
        if len(population) > 1:
            gdmean = mean(distances.distances.values())
            gdstdev = stdev(distances.distances.values())
            self.reporters.info(
                f'Mean genetic distance {gdmean:.3f}, standard deviation {gdstdev:.3f}')

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]

    def __getstate__(self):
        """Prepare species set for pickling by converting indexer to a picklable form."""
        state = self.__dict__.copy()
        # Convert the itertools.count object to an integer representing the next value
        if self.indexer is not None:
            state['_indexer_next_value'] = next(self.indexer)
            state['indexer'] = None
        else:
            state['_indexer_next_value'] = None
        return state

    def __setstate__(self, state):
        """Restore species set from pickled state, recreating indexer."""
        _indexer_next_value = state.pop('_indexer_next_value', None)
        self.__dict__.update(state)
        # Recreate the count object starting from the saved next value
        if _indexer_next_value is not None:
            self.indexer = count(_indexer_next_value)
        else:
            self.indexer = None
