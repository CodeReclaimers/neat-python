import random
import sys

from neat.indexer import Indexer
from neat.math_util import mean
from neat.six_util import iteritems, itervalues


class Species(object):
    def __init__(self, key, rep_id, representative):
        self.key = key
        self.representative = representative
        self.members = {rep_id: representative}
        self.age = 0

    def add(self, genome_id, genome):
        self.members[genome_id] = genome

    def get_fitnesses(self):
        return [m.fitness for m in itervalues(self.members)]

    def max_fitness(self):
        return max(self.get_fitnesses())

    def min_fitness(self):
        return min(self.get_fitnesses())

    def mean_fitness(self):
        return mean(self.get_fitnesses())

    def median_fitness(self):
        fitnesses = self.get_fitnesses()
        fitnesses.sort()
        return fitnesses[len(fitnesses) // 2]


class SpeciesSet(object):
    """
    Encapsulates the speciation scheme.
    """

    def __init__(self, config):
        self.indexer = Indexer(1)
        self.species = {}
        self.to_species = {}

    def speciate(self, config, population):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert type(population) is dict

        # Reset all species member lists.
        for s in itervalues(self.species):
            s.members.clear()
        self.to_species.clear()

        # Partition population into species based on genetic similarity.
        for key, individual in iteritems(population):
            # Find the species with the most similar representative.
            min_distance = sys.float_info.max
            closest_species = None
            closest_species_id = None
            for sid, s in iteritems(self.species):
                rep = s.representative
                distance, compatible = individual.distance(rep, config.genome_config)
                if compatible and distance < min_distance:
                    closest_species = s
                    closest_species_id = sid
                    min_distance = distance

            if closest_species is not None:
                closest_species.add(key, individual)
                self.to_species[key] = closest_species_id
            else:
                # No species is similar enough, create a new species for this individual.
                sid = self.indexer.get_next()
                self.species[sid] = Species(sid, key, individual)
                self.to_species[key] = sid

        # Only keep non-empty species.
        empty_species_ids = []
        for sid, s in iteritems(self.species):
            if not s.members:
                empty_species_ids.append(sid)

        for sid in empty_species_ids:
            del self.species[sid]

        # Select a random current member as the new representative.
        for s in itervalues(self.species):
            s.representative = random.choice(list(s.members.values()))

    def get_species_id(self, individual_id):
        return self.to_species[individual_id]

    def get_species(self, individual_id):
        sid = self.to_species[individual_id]
        return self.species[sid]