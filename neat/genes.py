"""Handles node and connection genes."""
import warnings
from random import random

from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute


# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.
# TODO: Evaluate using __slots__ for performance/memory usage improvement.


class BaseGene:
    """
    Handles functions shared by multiple types of genes (both node and connection),
    including crossover and calling mutation methods.
    """

    def __init__(self, key):
        self.key = key

    def __str__(self):
        attrib = ['key']
        if hasattr(self, 'innovation'):
            attrib.append('innovation')
        attrib += [a.name for a in self._gene_attributes]
        attrib = [f'{a}={getattr(self, a)}' for a in attrib]
        return f'{self.__class__.__name__}({", ".join(attrib)})'

    def __lt__(self, other):
        assert isinstance(self.key, type(other.key)), f"Cannot compare keys {self.key!r} and {other.key!r}"
        return self.key < other.key

    @classmethod
    def parse_config(cls, config, param_dict):
        pass

    @classmethod
    def get_config_params(cls):
        params = []
        if not hasattr(cls, '_gene_attributes'):
            setattr(cls, '_gene_attributes', getattr(cls, '__gene_attributes__'))
            warnings.warn(
                f"Class '{cls.__name__!s}' {cls!r} needs '_gene_attributes' not '__gene_attributes__'",
                DeprecationWarning)
        for a in cls._gene_attributes:
            params += a.get_config_params()
        return params

    @classmethod
    def validate_attributes(cls, config):
        for a in cls._gene_attributes:
            a.validate(config)

    def init_attributes(self, config):
        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config))

    def mutate(self, config):
        for a in self._gene_attributes:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))

    def copy(self):
        # Handle innovation number for connection genes
        if hasattr(self, 'innovation'):
            new_gene = self.__class__(self.key, innovation=self.innovation)
        else:
            new_gene = self.__class__(self.key)
        
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))

        return new_gene

    def crossover(self, gene2):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key
        
        # For connection genes, verify innovation numbers match
        # (they should represent the same historical mutation)
        if hasattr(self, 'innovation'):
            assert hasattr(gene2, 'innovation'), "Both genes must have innovation numbers"
            assert self.innovation == gene2.innovation, (
                f"Genes with same key must have same innovation number: "
                f"{self.innovation} vs {gene2.innovation}"
            )

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        if hasattr(self, 'innovation'):
            new_gene = self.__class__(self.key, innovation=self.innovation)
        else:
            new_gene = self.__class__(self.key)
        
        for a in self._gene_attributes:
            if random() > 0.5:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))
        
        # Implement the 75% disable rule from the NEAT paper:
        # If either parent has a disabled gene, there is a 75% chance
        # the offspring gene will be disabled.
        if hasattr(new_gene, 'enabled'):
            if not self.enabled or not gene2.enabled:
                if random() < 0.75:
                    new_gene.enabled = False

        return new_gene


# TODO: Should these be in the nn module?  iznn and ctrnn can have additional attributes.


class DefaultNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('bias'),
                        FloatAttribute('response'),
                        StringAttribute('activation', options=''),
                        StringAttribute('aggregation', options='')]

    def __init__(self, key):
        assert isinstance(key, int), f"DefaultNodeGene key must be an int, not {key!r}"
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.bias - other.bias) + abs(self.response - other.response)
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient


# TODO: Do an ablation study to determine whether the enabled setting is
# important--presumably mutations that set the weight to near zero could
# provide a similar effect depending on the weight range, mutation rate,
# and aggregation function. (Most obviously, a near-zero weight for the
# `product` aggregation function is rather more important than one giving
# an output of 1 from the connection, for instance!)
class DefaultConnectionGene(BaseGene):
    _gene_attributes = [FloatAttribute('weight'),
                        BoolAttribute('enabled')]

    def __init__(self, key, innovation=None):
        assert isinstance(key, tuple), f"DefaultConnectionGene key must be a tuple, not {key!r}"
        assert innovation is not None, "Innovation number is required for DefaultConnectionGene"
        assert isinstance(innovation, int), f"Innovation must be an int, not {type(innovation)}"
        BaseGene.__init__(self, key)
        self.innovation = innovation

    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient
    
    def __eq__(self, other):
        """Compare genes by innovation number."""
        if not isinstance(other, DefaultConnectionGene):
            return False
        return self.innovation == other.innovation
    
    def __hash__(self):
        """Hash by innovation number for use in sets/dicts."""
        return hash(self.innovation)
