import copy

import numpy as np

from neat.attributes import BoolAttribute, StringAttribute

from neat.genes import BaseGene
from neat.state_machine_attributes import ConditionsAttribute, BiasesAttribute, WeightsAttribute


class StateGene(BaseGene):
    """" Class representing the gene of a state in the state machine. """

    _gene_attributes = [BiasesAttribute('biases'),
                        WeightsAttribute('weights'),
                        StringAttribute('activation'),
                        StringAttribute('aggregation')]

    def __init__(self, key):
        assert isinstance(key, int), "StateGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        assert isinstance(other, StateGene)

        # Calculate the average difference in bias.
        avg_bias_difference = sum(np.abs(np.subtract(self.biases, other.biases))) / self.biases.size
        # Calculate the average difference in weights.
        avg_weight_difference = sum(sum(np.abs(np.subtract(self.weights, other.weights)))) / self.weights.size

        return config.compatibility_difference_coefficient * (avg_bias_difference + avg_weight_difference)

    def copy(self):
        state = StateGene(self.key)
        state.biases = np.array(self.biases)
        state.weights = np.array(self.weights)
        state.activation = self.activation
        state.aggregation = self.aggregation

        return state


class TransitionGene(BaseGene):
    """ Class representing the gene of a transition in the state machine. """

    _gene_attributes = [ConditionsAttribute('conditions'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), "TransitionGene key must be an tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        assert isinstance(other, TransitionGene)

        # TODO: add difference with maximal value for difference of condition being 1.
        # TODO: Track conditions (possibly)
        return config.compatibility_difference_coefficient * abs(len(other.conditions) - len(self.conditions))

    def copy(self):
        transition = TransitionGene(self.key)
        transition.conditions = copy.deepcopy(self.conditions)
        transition.enabled = self.enabled

        return transition
