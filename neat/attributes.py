"""Deals with the attributes (variable parameters) of genes"""
from copy import deepcopy
from random import choice, gauss, random, uniform, randint

from neat.config import ConfigParameter


# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.


class BaseAttribute:
    """Superclass for the type-specialized attribute subclasses, used by genes."""

    def __init__(self, name, **default_dict):
        self.name = name
        # Create instance-level copy to avoid sharing between instances (fixes GitHub issue #188)
        self._config_items = deepcopy(self.__class__._config_items)
        # TODO: implement a mechanism that allows us to detect and report unused configuration items.
        for n, default in default_dict.items():
            self._config_items[n] = [self._config_items[n][0], default]
        for n in self._config_items:
            setattr(self, n + "_name", self.config_item_name(n))

    def config_item_name(self, config_item_base_name):
        return f"{self.name}_{config_item_base_name}"

    def get_config_params(self):
        return [ConfigParameter(self.config_item_name(n), ci[0], ci[1])
                for n, ci in self._config_items.items()]


class FloatAttribute(BaseAttribute):
    """
    Class for floating-point numeric attributes,
    such as the response of a node or the weight of a connection.
    """
    _config_items = {"init_mean": [float, None],
                     "init_stdev": [float, None],
                     "init_type": [str, 'gaussian'],
                     "replace_rate": [float, None],
                     "mutate_rate": [float, None],
                     "mutate_power": [float, None],
                     "max_value": [float, None],
                     "min_value": [float, None]}

    def clamp(self, value, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        return max(min(value, max_value), min_value)

    def init_value(self, config):
        mean = getattr(config, self.init_mean_name)
        stdev = getattr(config, self.init_stdev_name)
        init_type = getattr(config, self.init_type_name).lower()

        if ('gauss' in init_type) or ('normal' in init_type):
            return self.clamp(gauss(mean, stdev), config)

        if 'uniform' in init_type:
            min_value = max(getattr(config, self.min_value_name),
                            (mean - (2 * stdev)))
            max_value = min(getattr(config, self.max_value_name),
                            (mean + (2 * stdev)))
            return uniform(min_value, max_value)

        raise RuntimeError(f"Unknown init_type {getattr(config, self.init_type_name)!r} for {self.init_type_name!s}")

    def mutate_value(self, value, config):
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        mutate_rate = getattr(config, self.mutate_rate_name)

        r = random()
        if r < mutate_rate:
            mutate_power = getattr(config, self.mutate_power_name)
            return self.clamp(value + gauss(0.0, mutate_power), config)

        replace_rate = getattr(config, self.replace_rate_name)

        if r < replace_rate + mutate_rate:
            return self.init_value(config)

        return value

    def validate(self, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        if max_value < min_value:
            raise RuntimeError("Invalid min/max configuration for {self.name}")


class IntegerAttribute(BaseAttribute):
    """
    Class for integer numeric attributes.
    """
    _config_items = {"replace_rate": [float, None],
                     "mutate_rate": [float, None],
                     "mutate_power": [float, None],
                     "max_value": [int, None],
                     "min_value": [int, None]}

    def clamp(self, value, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        return max(min(value, max_value), min_value)

    def init_value(self, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        return randint(min_value, max_value)

    def mutate_value(self, value, config):
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        mutate_rate = getattr(config, self.mutate_rate_name)

        r = random()
        if r < mutate_rate:
            mutate_power = getattr(config, self.mutate_power_name)
            return self.clamp(value + int(round(gauss(0.0, mutate_power))), config)

        replace_rate = getattr(config, self.replace_rate_name)

        if r < replace_rate + mutate_rate:
            return self.init_value(config)

        return value

    def validate(self, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        if max_value < min_value:
            raise RuntimeError("Invalid min/max configuration for {self.name}")


class BoolAttribute(BaseAttribute):
    """Class for boolean attributes such as whether a connection is enabled or not."""
    _config_items = {"default": [str, None],
                     "mutate_rate": [float, None],
                     "rate_to_true_add": [float, 0.0],
                     "rate_to_false_add": [float, 0.0]}

    def init_value(self, config):
        default = str(getattr(config, self.default_name)).lower()

        if default in ('1', 'on', 'yes', 'true'):
            return True
        elif default in ('0', 'off', 'no', 'false'):
            return False
        elif default in ('random', 'none'):
            return bool(random() < 0.5)

        raise RuntimeError(f"Unknown default value {default!r} for {self.name!s}")

    def mutate_value(self, value, config):
        mutate_rate = getattr(config, self.mutate_rate_name)

        if value:
            mutate_rate += getattr(config, self.rate_to_false_add_name)
        else:
            mutate_rate += getattr(config, self.rate_to_true_add_name)

        if mutate_rate > 0:
            r = random()
            if r < mutate_rate:
                # NOTE: we choose a random value here so that the mutation rate has the
                # same exact meaning as the rates given for the string and bool
                # attributes (the mutation operation *may* change the value but is not
                # guaranteed to do so).
                return random() < 0.5

        return value

    def validate(self, config):
        default = str(getattr(config, self.default_name)).lower()
        if default not in ('1', 'on', 'yes', 'true', '0', 'off', 'no', 'false', 'random', 'none'):
            raise RuntimeError("Invalid default value for {self.name}")


class StringAttribute(BaseAttribute):
    """
    Class for string attributes such as the aggregation function of a node,
    which are selected from a list of options.
    """
    _config_items = {"default": [str, 'random'],
                     "options": [list, None],
                     "mutate_rate": [float, None]}

    def init_value(self, config):
        default = getattr(config, self.default_name)

        if default.lower() in ('none', 'random'):
            options = getattr(config, self.options_name)
            return choice(options)

        return default

    def mutate_value(self, value, config):
        mutate_rate = getattr(config, self.mutate_rate_name)

        if mutate_rate > 0:
            r = random()
            if r < mutate_rate:
                options = getattr(config, self.options_name)
                return choice(options)

        return value

    def validate(self, config):
        default = getattr(config, self.default_name)
        if default not in ('none', 'random'):
            options = getattr(config, self.options_name)
            if default not in options:
                raise RuntimeError(f'Invalid initial value {default} for {self.name}')
            assert default in options
