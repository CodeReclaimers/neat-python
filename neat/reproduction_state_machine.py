import math

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

        # Get the best performing species
        best_performing_species = self.get_best_performing_species(remaining_species)

        print('best performing is: ' + str(best_performing_species))

        for species, stagnant in zip(remaining_species, stagnant_species):

            old_members = list(iteritems(species.members))
            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            spawn = len(species.members)
            spawned = 0
            species.members = {}  # Reset the set of species.
            children = {}

            # Transfer elites to new generation.
            elites = []
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    children[i] = m
                    elites.append(i)
                    spawned += 1

            repro_cutoff = self.calculated_cutoff(len(old_members))
            old_members = old_members[:repro_cutoff]

            # Generate the new children.
            for _ in range(spawn - spawned):
                gid, child = self.generate_child(old_members, config)
                children[gid] = child

            # If the current species is stagnant add a state to 1/2 of the state's population.
            if stagnant:

                updated_children = math.floor(spawn / 2)
                spawn_all = False

                if spawn - updated_children <= self.reproduction_config.min_species_size:
                    # If species becomes to small and is not the best performing species, upgrade all.
                    spawn_all = True

                for gid in children:

                    if (updated_children > 0 and gid not in elites) or spawn_all:
                        self.update_child(gid, children, species, config, best_performing_species)
                        updated_children -= 1

            new_population.update(children)

        assert len(new_population) == pop_size
        return new_population

    def update_child(self, gid, children, species, config, best_performing_species):
        # If the species has to much reproduction, then create a new child with as number of states,
        # so much as the best performing species.
        if species.key >= self.reproduction_config.max_num_states:
            children[gid] = config.genome_type(gid)
            children[gid].configure_new(config.genome_config, best_performing_species)
        else:
            children[gid].mutate_add_state(config.genome_config)

    @staticmethod
    def get_best_performing_species(remaining_species):
        """ This function returns the index of the best performing species in the species list. """
        best_performing_species = -1
        best_fitness = -float('inf')
        for remaining_species in remaining_species:
            if remaining_species.fitness > best_fitness:
                best_performing_species = remaining_species.key
                best_fitness = remaining_species.fitness

        return best_performing_species


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
