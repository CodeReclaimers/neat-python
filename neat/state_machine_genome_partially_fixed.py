import pickle

from neat.config import ConfigParameter
from neat.state_machine_genes import StateGene, TransitionGene
from neat.state_machine_genome import StateMachineGenomeConfig, StateMachineGenome

"""
This file describes a state machine genome, where part of the state machine is already fixed and cannot be altered.
For example, The states are already fixed and only the connections between the states can be alterd.
Or the connections are already fixed and only the weights of the states can be altered.

for the fixed_section keyword there are 3 possible keywords: states, transitions, layout which means whether 
the states, the transitions respectively the layout of the state machine is fixed.
Fixing the states or the transitions automatically means fixing the layout.
"""


class StateMachineGenomeFixedConfig(StateMachineGenomeConfig):

    def __init__(self, params):
        super().__init__(params)

        additional_params = [ConfigParameter('genome_source', str),
                             ConfigParameter('fixed_section', str)]
        self._params.extend(additional_params)

        for p in additional_params:
            setattr(self, p.name, p.interpret(params))


class StateMachineGenomeFixed(StateMachineGenome):

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = StateGene
        param_dict['connection_gene_type'] = TransitionGene
        return StateMachineGenomeFixedConfig(param_dict)

    def configure_new(self, config):
        """ Override this function, since now we have to take into account that part of the genome is fixed."""
        # Load the example gene.
        genome = pickle.load(open(config.genome_source, "rb"))

        if config.fixed_section == 'states':
            # Copy the states from the given genome.
            for key, state in genome.states.items():
                self.states[key] = state.copy()

        elif config.fixed_section == 'transitions':
            # Collect all the states that occur in the given state machines.
            state_keys = set()
            for begin, end in genome.transitions:
                state_keys.add(begin)
                state_keys.add(end)

            # Randomly initialize the states since those need to be evolved.
            for key in state_keys:
                state = self.create_state(config, key)
                self.states[key] = state

            # Copy the transitions from the given genome.
            for key, transition in genome.transitions.items():
                self.transitions[key] = transition.copy()

            assert len(self.transitions) == len(genome.transitions)

        elif config.fixed_section == 'layout':
            # Randomly initialise the given states and
            for key in genome.states:
                self.states[key] = self.create_state(config, key)

            for begin, end in genome.transitions:
                self.transitions[(begin, end)] = self.create_transition(config, begin, end)

        else:
            print('Invalid config parameter, fixed_section: ' + config.fixed_section)
            print('Valid values for this parameter are: states, transitions, layout')

    def mutate_states(self, config):
        """ Override mutate_states to prevent state mutations if that section is fixed."""
        if config.fixed_section != 'states':

            # Mutate node genes (bias, response, etc.).
            for ng in self.states.values():
                ng.mutate(config)

        # Note that regarding of the fixed section the number of states is always fixed, since it does not make sense
        # to change them if the layout, or the number of transitions is fixed.

    def mutate_transitions(self, config):
        """ Override mutate_transitions to prevent transition mutations if that section is fixed."""
        if config.fixed_section not in ['transitions', 'layout']:
            # Only allow to mutate transitions if the number of states is fixed.
            super().mutate_transitions(config)
