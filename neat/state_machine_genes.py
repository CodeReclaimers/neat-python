from neat.attributes import BoolAttribute

from neat.genes import BaseGene
from neat.state_machine_attributes import ConditionsAttribute, BiasesAttribute, WeightsAttribute


class StateGene(BaseGene):
    """" Class representing the gene of a state in the state machine. """

    _gene_attributes = [BiasesAttribute('biases'),
                        WeightsAttribute('weights')]

    def __init__(self, key):
        assert isinstance(key, int), "StateGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        assert isinstance(other, StateGene)

        self.biases.difference()

        return


class TransitionGene(BaseGene):
    """ Class representing the gene of a transition in the state machine. """

    _gene_attributes = [ConditionsAttribute('conditions'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), "TransitionGene key must be an tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        assert isinstance(other, TransitionGene)

        return 0
