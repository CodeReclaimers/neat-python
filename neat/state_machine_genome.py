from random import random, choice

from neat.config import ConfigParameter

from neat.aggregations import AggregationFunctionSet
from neat.six_util import iteritems

from neat.activations import ActivationFunctionSet
from neat.state_machine_genes import StateGene, TransitionGene


class StateMachineGenomeConfig(object):
    """ Class containing the parameters required to actually build the config file."""
    def __init__(self, params):

        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()

        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('num_initial_states', int),
                        ConfigParameter('max_num_states', int, 1000),   # Note the default.
                        ConfigParameter('state_add_prob', float),
                        ConfigParameter('state_delete_prob', float),
                        ConfigParameter('transition_add_prob', float),
                        ConfigParameter('transition_delete_prob', float),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_difference_coefficient', float)]

        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        self.node_indexer = self.num_initial_states

    def get_new_node_key(self):

        new_node = self.node_indexer
        self.node_indexer += 1

        return new_node


class StateMachineGenome(object):
    """
    A genome for the state machine with simple neural networks in it.
        Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        state: State of the state machine.
        transition: Transition from one state machine to another.
        key: Identifier for an object, unique within the set of similar objects.
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = StateGene
        param_dict['connection_gene_type'] = TransitionGene
        return StateMachineGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.states = {}
        self.transitions = {}

        # Fitness results.
        self.fitness = None

    def configure_new(self, config):
        """" Create a simple state machine without any outgoing states. """
        for i in range(config.num_initial_states):
            self.states[i] = self.create_state(config, i)

    @staticmethod
    def create_state(config, state_key):

        state = StateGene(state_key)
        state.init_attributes(config)
        return state

    @staticmethod
    def create_transition(config, begin_key, end_key):

        transition = TransitionGene((begin_key, end_key))
        transition.init_attributes(config)
        return transition

    def clone(self, genome):
        """ This function clones the given genome in the current genome. """
        for key, connection in iteritems(genome.transitions):
            self.transitions[key] = connection.copy()

        for key, state in iteritems(genome.states):
            self.states[key] = state.copy()

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))

        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in iteritems(parent1.transitions):
            cg2 = parent2.transitions.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.transitions[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.transitions[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.states
        parent2_set = parent2.states

        for key, ng1 in iteritems(parent1_set):
            ng2 = parent2_set.get(key)
            assert key not in self.states
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.states[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.states[key] = ng1.crossover(ng2)

    def mutate(self, config):
        """ Mutates this genome. """

        self.mutate_states(config)
        self.mutate_transitions(config)

    def mutate_states(self, config):
        """ This function mutates the states of the genome. """

        if len(self.states) < config.max_num_states and random() < config.state_add_prob:
            self.mutate_add_state(config)

        if random() < config.state_delete_prob:
            self.mutate_delete_state(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.states.values():
            ng.mutate(config)

    def mutate_transitions(self, config):
        """ This function mutates the transitions of the genome."""

        if random() < config.transition_add_prob:
            self.mutate_add_transition(config)

        if random() < config.transition_delete_prob:
            self.mutate_delete_transition()

        # Mutate connection genes.
        for cg in self.transitions.values():
            cg.mutate(config)

    def mutate_add_state(self, config):

        new_state = self.create_state(config, config.get_new_node_key())

        # pick 2 random nodes make an incoming and an outgoing transition.
        incoming_state = choice(list(self.states.values()))
        outgoing_state = choice(list(self.states.values()))

        t1 = self.create_transition(config, incoming_state.key, new_state.key)
        t2 = self.create_transition(config, new_state.key, outgoing_state.key)

        # Enter values in dictionary.
        self.states[new_state.key] = new_state
        self.transitions[t1.key] = t1
        self.transitions[t2.key] = t2

    def mutate_add_transition(self, config):
        # Pick two random states which need to be connected.
        begin_state_key = choice(list(self.states.values())).key
        end_state_key = choice(list(self.states.values())).key

        if begin_state_key != end_state_key:  # Raise exception when self-loop is introduced.

            key = (begin_state_key, end_state_key)

            # Check whether there is a connection already, if so enable otherwise create one.
            if key in self.transitions:
                self.transitions[key].enabled = True
            else:
                t = self.create_transition(config, begin_state_key, end_state_key)
                self.transitions[t.key] = t

    def mutate_delete_transition(self):
        """" Removes one of the transitions currently available. """
        if self.transitions:
            key = choice(list(self.transitions.keys()))
            del self.transitions[key]

    def mutate_delete_state(self, config):
        # Ensure at least one state survives.
        if len(self.states) > 1:
            # Select state to delete.
            del_key = choice(list(self.states.keys()))

            # Make sure that the initial state stays, since that is where the algorithm starts with.
            if del_key == 0:
                return

            # Remove all transitions to and from this state.
            transitions_to_delete = set()
            for k, _ in self.transitions.items():
                if del_key in k:
                    transitions_to_delete.add(k)

            for key in transitions_to_delete:
                del self.transitions[key]

            # Remove state.
            del self.states[del_key]

    def distance(self, other, config):
        return self.distance_num_node_difference(other, config)

    def distance_num_node_difference(self, other, config):
        return abs(len(self.states) - len(other.states))

    def advanced_distance(self, other, config):
        difference_distance = 0

        similarity_count = 0
        for key, state in self.states.items():
            if key in other.states:
                difference_distance += state.distance(other.states[key], config)
                similarity_count += 1

        for key, transition in self.transitions.items():
            if key in other.transitions:
                difference_distance += transition.distance(other.transitions[key], config)

        # TODO: add transition disjoint distance.
        num_disjoint_states = len(self.states) + len(other.states) - (similarity_count * 2)

        return config.compatibility_disjoint_coefficient * num_disjoint_states + difference_distance

    def size(self):
        return len(self.states), len(self.transitions)

    def __str__(self):

        s = "Key: {0}\nFitness: {1}\nNodes:".format(self.key, self.fitness)
        for k, ng in iteritems(self.states):
            s += "\n\t{0} {1!s}".format(k, ng)
        s += "\nConnections:"
        transitions = list(self.transitions.values())
        transitions.sort()
        for c in transitions:
            s += "\n\t" + str(c)
        return s
