from random import randint, random

from neat.state_machine_genome import StateMachineGenome


class StateMachineFullGenome(StateMachineGenome):
    """
    This class describes a state machine that does not add or remove connections, and performs crossover within
    genomes of the same size.
    Furthermore this genome only works on fully connected state machines.
    """

    def configure_new(self, config):
        """" Create a fully connected state machine with given number of states. """
        for i in range(config.num_initial_states):
            self.states[i] = self.create_state(config, i)

        # Create a connection from and to everywhere
        for i in range(config.num_initial_states):
            for j in range(config.num_initial_states):
                if i != j:
                    self.transitions[(i, j)] = self.create_transition(config, i, j)

    def configure_crossover(self, genome1, genome2, config):
        """
        The crossover operator takes a random subset of states from the parent and the remaining states from the
        other parent. Connections between the subset states are copied from the parent, connections that cross subset
        are randomly selected.
        Assumption genomes have the same number of states, this is true because they belong to the same species.
        """
        selected_node_keys = random_subset(genome1.states)

        # Copy selected nodes from genome 1
        for nk in selected_node_keys:
            self.states[nk] = genome1.states[nk]

            for nk2 in selected_node_keys:
                if nk != nk2:
                    self.transitions[(nk, nk2)] = genome1.transitions[(nk, nk2)].copy()

        # Copy other nodes from genome 2
        not_selected_node_keys = genome1.keys().difference(selected_node_keys)
        for nk in not_selected_node_keys:
            self.states[nk] = genome2.states[nk]

            for nk2 in not_selected_node_keys:
                if nk != nk2:
                    self.transitions[(nk, nk2)] = genome2.transitions[(nk, nk2)].copy()

        # Randomly select other transitions.
        for nk in selected_node_keys:
            for nk2 in not_selected_node_keys:

                # Do connection from fst node to snd node
                selected_transition = genome1.transitions[(nk, nk2)]
                if random() < 0.5:
                    selected_transition = genome2.transitions[(nk, nk2)]

                self.transitions[(nk, nk2)] = selected_transition.copy()

                # Do connection from snd node to fst node.
                selected_transition = genome1.transitions[(nk2, nk)]
                if random() < 0.5:
                    selected_transition = genome2.transitions[(nk2, nk)]

                self.transitions[(nk2, nk)] = selected_transition.copy()

    def mutate_states(self, config):

        # Do not add or remove states.
        for ng in self.states.values():
            ng.mutate(config)

    def mutate_transitions(self, config):

        # Do not add or remove transitions.
        for cg in self.transitions.values():
            cg.mutate(config)

    def mutate_add_state(self, config):

        new_state = self.create_state(config, config.get_new_node_key())

        # Add incoming and outgoing transitions to each state.
        for state_key in self.states:

            t1 = self.create_transition(config, state_key, new_state.key)
            t2 = self.create_transition(config, new_state.key, state_key)

            self.transitions[t1.key] = t1
            self.transitions[t2.key] = t2

        self.states[new_state.key] = new_state

    def distance(self, other, config):
        """ Distance as the difference in number of states. """
        return abs(len(self.states) - len(other.states))


def random_subset(s):
    """ This function returns a random subset of the given set."""
    out = set()
    for el in s:
        # random coin flip
        if randint(0, 1) == 0:
            out.add(el)
    return out
