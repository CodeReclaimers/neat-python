"""Deals with the attributes (variable parameters) of genes"""
from __future__ import annotations
from random import choice, gauss, random, uniform
from neat.config import ConfigParameter
from typing import List, Final, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neat.genome import DefaultGenomeConfig


# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.


class BaseAttribute(object):
    """Superclass for the type-specialized attribute subclasses, used by genes.

    init_mean, init_strdev, init_type, mutate_rate
    など特定のGeneに関する設定プロパティをgetattrで動的にメンバに追加する
    遺伝子に利用される
    """
    _config_items: Dict[str, List[Any]]

    def __init__(self, name: str, **default_dict):
        self.name: str = name
        self.init_mean_name: str = ""
        self.init_stdev_name: str = ""
        self.init_type_name: str = ""
        self.max_value_name: str = ""
        self.min_value_name: str = ""
        self.mutate_power_name: str = ""
        self.mutate_rate_name: str = ""
        self.replace_rate_name: str = ""
        self.default_name: str = ""
        self.options_name: str = ""
        self.rate_to_false_add_name: str = ""
        self.rate_to_true_add_name: str = ""

        for n, default in default_dict.items():
            self._config_items[n] = [self._config_items[n][0], default]
        for n in self._config_items:
            setattr(self, n + "_name", self.config_item_name(n))

    def config_item_name(self, config_item_base_name: str) -> str:
        return "{0}_{1}".format(self.name, config_item_base_name)

    def get_config_params(self) -> List[ConfigParameter]:
        return [ConfigParameter(self.config_item_name(n),
                                self._config_items[n][0],
                                self._config_items[n][1])
                for n in self._config_items]

    def init_value(self, config: DefaultGenomeConfig):
        pass

    def mutate_value(self, v, config: DefaultGenomeConfig):
        pass


class FloatAttribute(BaseAttribute):
    """
    Class for numeric attributes,
    such as the response of a node or the weight of a connection.

    数値的な設定値を管理するクラス init_meanなど
    """
    _config_items: Dict[str, List[Any]] = {"init_mean": [float, None],
                                           "init_stdev": [float, None],
                                           "init_type": [str, 'gaussian'],
                                           "replace_rate": [float, None],
                                           "mutate_rate": [float, None],
                                           "mutate_power": [float, None],
                                           "max_value": [float, None],
                                           "min_value": [float, None]}

    def clamp(self, value: float, config: DefaultGenomeConfig) -> float:
        min_value: float = getattr(config, self.min_value_name)
        max_value: float = getattr(config, self.max_value_name)
        return max(min(value, max_value), min_value)

    def init_value(self, config: DefaultGenomeConfig) -> float:
        mean: float = getattr(config, self.init_mean_name)
        stdev: float = getattr(config, self.init_stdev_name)
        init_type: str = getattr(config, self.init_type_name).lower()

        if ('gauss' in init_type) or ('normal' in init_type):
            return self.clamp(gauss(mean, stdev), config)

        if 'uniform' in init_type:
            min_value = max(getattr(config, self.min_value_name),
                            (mean - (2 * stdev)))
            max_value = min(getattr(config, self.max_value_name),
                            (mean + (2 * stdev)))
            return uniform(min_value, max_value)

        raise RuntimeError("Unknown init_type {!r} for {!s}".format(getattr(config,
                                                                            self.init_type_name),
                                                                    self.init_type_name))

    def mutate_value(self, value: float, config: DefaultGenomeConfig) -> float:
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        # ConfigとAttributes（設定ファイル）を元に変化させた値を返す
        mutate_rate: float = getattr(config, self.mutate_rate_name)

        r: float = random()
        if r < mutate_rate:
            mutate_power = getattr(config, self.mutate_power_name)
            return self.clamp(value + gauss(0.0, mutate_power), config)

        replace_rate: float = getattr(config, self.replace_rate_name)

        if r < replace_rate + mutate_rate:
            return self.init_value(config)

        return value

    def validate(self, config: DefaultGenomeConfig):  # pragma: no cover
        pass


class BoolAttribute(BaseAttribute):
    """Class for boolean attributes such as whether a connection is enabled or not."""
    _config_items: Dict[str, List[Any]] = {"default": [str, None],
                                           "mutate_rate": [float, None],
                                           "rate_to_true_add": [float, 0.0],
                                           "rate_to_false_add": [float, 0.0]}

    def init_value(self, config: DefaultGenomeConfig) -> bool:
        default: str = str(getattr(config, self.default_name)).lower()

        if default in ('1', 'on', 'yes', 'true'):
            return True
        elif default in ('0', 'off', 'no', 'false'):
            return False
        elif default in ('random', 'none'):
            return bool(random() < 0.5)

        raise RuntimeError("Unknown default value {!r} for {!s}".format(default,
                                                                        self.name))

    def mutate_value(self, value: bool, config: DefaultGenomeConfig) -> bool:
        mutate_rate: float = getattr(config, self.mutate_rate_name)

        if value:
            mutate_rate += getattr(config, self.rate_to_false_add_name)
        else:
            mutate_rate += getattr(config, self.rate_to_true_add_name)

        if mutate_rate > 0:
            r: float = random()
            if r < mutate_rate:
                # NOTE: we choose a random value here so that the mutation rate has the
                # same exact meaning as the rates given for the string and bool
                # attributes (the mutation operation *may* change the value but is not
                # guaranteed to do so).
                return random() < 0.5

        return value

    def validate(self, config: DefaultGenomeConfig):  # pragma: no cover
        pass


class StringAttribute(BaseAttribute):
    """
    Class for string attributes such as the aggregation function of a node,
    which are selected from a list of options.
    """
    _config_items = {"default": [str, 'random'],
                     "options": [list, None],
                     "mutate_rate": [float, None]}

    def init_value(self, config: DefaultGenomeConfig) -> str:
        default: str = getattr(config, self.default_name)

        if default.lower() in ('none', 'random'):
            options = getattr(config, self.options_name)
            return choice(options)

        return default

    def mutate_value(self, value: str, config: DefaultGenomeConfig) -> str:
        mutate_rate: float = getattr(config, self.mutate_rate_name)

        if mutate_rate > 0:
            r: float = random()
            if r < mutate_rate:
                options = getattr(config, self.options_name)
                return choice(options)

        return value

    def validate(self, config: DefaultGenomeConfig):  # pragma: no cover
        pass
