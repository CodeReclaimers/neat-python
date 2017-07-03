from random import random
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute

# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.
# TODO: Evaluate using __slots__ for performance/memory usage improvement.


class BaseGene(object):
    def __init__(self, key):
        self.key = key

    def __str__(self):
        attrib = ['key'] + [a.name for a in self.__gene_attributes__]
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attrib]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))

    def __lt__(self, other):
        return self.key < other.key

    @classmethod
    def parse_config(cls, config, param_dict):
        pass

    @classmethod
    def get_config_params(cls):
        params = []
        for a in cls.__gene_attributes__:
            params += a.get_config_params()
        return params

    def init_attributes(self, config):
        for a in self.__gene_attributes__:
            setattr(self, a.name, a.init_value(config))

    def mutate(self, config):
        for a in self.__gene_attributes__:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))

    def copy(self):
        new_gene = self.__class__(self.key)
        for a in self.__gene_attributes__:
            setattr(new_gene, a.name, getattr(self, a.name))

        return new_gene

    def crossover(self, gene2):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        new_gene = self.__class__(self.key)
        for a in self.__gene_attributes__:
            if random() > 0.5:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))

        return new_gene


# TODO: Should these be in the nn module?  iznn and ctrnn can have additional attributes.


class DefaultNodeGene(BaseGene):
    __gene_attributes__ = [FloatAttribute('bias'),
                           FloatAttribute('response'),
                           StringAttribute('activation', 'sigmoid'),
                           StringAttribute('aggregation', 'sum')]

    def distance(self, other, config):
        d = abs(self.bias - other.bias) + abs(self.response - other.response)
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient


# TODO: Do an ablation study to determine whether the enabled setting is
# important--presumably mutations that set the weight to near zero could
# provide a similar effect depending on the weight range and mutation rate.
class DefaultConnectionGene(BaseGene):
    __gene_attributes__ = [FloatAttribute('weight'),
                           BoolAttribute('enabled')]

    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient

