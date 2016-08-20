import random

from neat.indexer import Indexer


class Species(object):
    """ A collection of genetically similar individuals."""

    def __init__(self, representative, ID):
        assert type(ID) == int
        self.representative = representative
        self.ID = ID
        self.age = 0
        self.members = []

        self.add(representative)

    def add(self, individual):
        individual.species_id = self.ID
        self.members.append(individual)


class SpeciesSet(object):
    """
    Encapsulates the speciation scheme.
    """
    def __init__(self, config):
        self.config = config
        self.indexer = Indexer(1)
        self.species = []

    def speciate(self, population):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        for individual in population:
            # Find the species with the most similar representative.
            min_distance = None
            closest_species = None
            for s in self.species:
                distance = individual.distance(s.representative)
                if distance < self.config.compatibility_threshold \
                        and (min_distance is None or distance < min_distance):
                    closest_species = s
                    min_distance = distance

            if closest_species is not None:
                closest_species.add(individual)
            else:
                # No species is similar enough, create a new species for this individual.
                self.species.append(Species(individual, self.indexer.get_next()))

        # Only keep non-empty species.
        self.species = [s for s in self.species if s.members]

        # Select a random current member as the new representative.
        for s in self.species:
            s.representative = random.choice(s.members)
