"""Handles node and connection genes."""
from __future__ import annotations

import warnings

from random import random
from typing import List, Optional, Dict, Set, Tuple, Union, Final, TYPE_CHECKING
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute, BaseAttribute
from neat.config import ConfigParameter

if TYPE_CHECKING:
    from neat.genome import DefaultGenomeConfig


# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.
# TODO: Evaluate using __slots__ for performance/memory usage improvement.


class BaseGene(object):
    """
    Handles functions shared by multiple types of genes (both node and connection),
    including crossover and calling mutation methods.

    ノード遺伝子と接続遺伝子のベースクラス
    """

    # これもまずかったら即消せ
    # Geneに関するユーザ設定値
    _gene_attributes: List[BaseAttribute]

    def __init__(self, key: Union[int, Tuple[int, int]]) -> None:
        self.key: Final[Union[int, Tuple[int, int]]] = key

        # add init (まずかったら即消せ)
        self.bias: float = 0
        self.weight: float = 0
        self.response: float = 0
        self.enabled: bool = True
        self.activation: str = ""
        self.aggregation: str = ""

    def __str__(self) -> str:
        attrib = ['key'] + [a.name for a in self._gene_attributes]
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attrib]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))

    def __lt__(self, other: BaseGene) -> bool:
        assert isinstance(self.key, type(other.key)), "Cannot compare keys {0!r} and {1!r}".format(self.key, other.key)
        return self.key < other.key

    @classmethod
    def parse_config(cls, config, param_dict):
        pass

    @classmethod
    def get_config_params(cls) -> List[ConfigParameter]:
        """
        ノードに関するConfigParameterのリストを返す
        ConfigParameterがある一つのアイテム（本当にひとつ）のユーザパラメータを扱うレコード
        たとえば、`bias_mutate_power`などが一つのConfigParameter
        このとき、返すListに値自体は設定されていない
        """
        params: List[ConfigParameter] = []
        if not hasattr(cls, '_gene_attributes'):
            setattr(cls, '_gene_attributes', getattr(cls, '__gene_attributes__'))
            warnings.warn(
                "Class '{!s}' {!r} needs '_gene_attributes' not '__gene_attributes__'".format(
                    cls.__name__, cls),
                DeprecationWarning)
        for a in cls._gene_attributes:
            params += a.get_config_params()
        # 遺伝子に関わるConfigParameterがすべて入る
        return params

    def init_attributes(self, config: DefaultGenomeConfig):
        """
        geneに関するweight, enabledなどのみを初期化する
        """
        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config))

    def mutate(self, config: DefaultGenomeConfig):
        for a in self._gene_attributes:
            v: float = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))

    def copy(self) -> BaseGene:
        new_gene: BaseGene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))

        return new_gene

    def crossover(self, gene2: BaseGene) -> BaseGene:
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        new_gene: BaseGene = self.__class__(self.key)
        for a in self._gene_attributes:
            if random() > 0.5:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))

        return new_gene


# TODO: Should these be in the nn module?  iznn and ctrnn can have additional attributes.


class DefaultNodeGene(BaseGene):
    _gene_attributes: List[BaseAttribute] = [FloatAttribute('bias'),
                                             FloatAttribute('response'),
                                             StringAttribute('activation', options='sigmoid'),
                                             StringAttribute('aggregation', options='sum')]

    def __init__(self, key: int):
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other: DefaultNodeGene, config: DefaultGenomeConfig) -> float:
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
    _gene_attributes: List[BaseAttribute] = [FloatAttribute('weight'),
                                             BoolAttribute('enabled')]

    def __init__(self, key: Tuple[int, int]):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other: DefaultConnectionGene, config: DefaultGenomeConfig) -> float:
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient
