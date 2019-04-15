import random

from neat import DefaultReproduction


class ReproductionMutationOnly(DefaultReproduction):
    """ This class does reproduction with only mutations and no crossover. """

    def generate_child(self, old_members, config):
        """ This function generates a child using only mutation."""
        # Select a parent
        parent_id, parent = random.choice(old_members)

        # Generate a child and clone and mutate it.
        gid = next(self.genome_indexer)
        child = config.genome_type(gid)
        child.clone(parent)
        child.mutate(config.genome_config)
        self.ancestors[gid] = parent_id
        return gid, child
