from neat import DefaultSpeciesSet
from neat.config import ConfigParameter, DefaultClassConfig
from neat.reproduction_mutation_only import ReproductionMutationOnly
from neat.six_util import itervalues, iteritems
from neat.species import Species


class ReproductionStateMachineOnly(ReproductionMutationOnly):
    """ This class reproduces a state machine given the idea of starting with minimal states and
    extending the number of states when species stagnates.
    Each species has its own number of states, and the number of states determines how well a species performs.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('max_num_states', int, 5),
                                   ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2)
                                   ])

    def reproduce(self, config, species, pop_size, generation):

        # Get a list of the stagnated species.
        all_fitnesses = []
        remaining_species = []
        stagnant_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)

            remaining_species.append(stag_s)
            stagnant_species.append(stagnant)
            all_fitnesses.extend(m.fitness for m in itervalues(stag_s.members))

        self.calculate_adjusted_fitnesses(remaining_species, all_fitnesses)

        new_population = {}

        # TODO: if the max_num_states is reached and species is stagnated add species to best performing species
        # TODO: Make species key be the number of states.
        # TODO: Best performing species cannot disappear and stays at min_species_size

        for species, stagnant in zip(remaining_species, stagnant_species):

            old_members = list(iteritems(species.members))

            spawned = len(species.members)
            species.members = {}  # Reset the set of species.

            for _ in range(spawned):
                gid, child = self.generate_child(old_members, config)
                new_population[gid] = child

        print(len(new_population))
        print(pop_size)
        assert len(new_population) == pop_size
        return new_population


class StateSeparatedSpeciesSet(DefaultSpeciesSet):
    """ This class represents a species set, where the species are separated by number of states. """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict, [])

    def speciate(self, config, population, generation):

        assert isinstance(population, dict)

        self.genome_to_species = {}

        # Add all the genomes to the right species set.
        for gid, genome in iteritems(population):

            num_states = len(genome.states)

            if num_states not in self.species:
                self.species[num_states] = Species(num_states, generation)

            self.species[num_states].members[gid] = genome
            self.genome_to_species[gid] = num_states

        # Remove empty species sets.
        empty = []
        for sid, species in iteritems(self.species):
            if not species.members:
                empty.append(sid)

        for sid in empty:
            self.species.pop(sid)
            self.reporters.info('Removing species with {0} states from species set'.format(sid))

        self.reporters.info('Speciated into {0} species'.format(len(self.species)))
