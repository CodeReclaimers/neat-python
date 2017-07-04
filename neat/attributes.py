"""Deals with the attributes (variable parameters) of genes"""
from random import choice, gauss, random, uniform
from neat.config import ConfigParameter
from neat.six_util import iterkeys, iteritems

# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.


class BaseAttribute(object):
    def __init__(self, name, **default_dict):
        self.name = name
        for n, default in iteritems(default_dict):
            self.__config_items__[n] = [self.__config_items__[n][0], default]
        for n in iterkeys(self.__config_items__):
            setattr(self, n + "_name", self.config_item_name(n))

    def config_item_name(self, config_item_base_name):
        return "{0}_{1}".format(self.name, config_item_base_name)

    def get_config_params(self):
        return [ConfigParameter(self.config_item_name(n),
                                self.__config_items__[n][0],
                                self.__config_items__[n][1]) for n in iterkeys(self.__config_items__)]

class FloatAttribute(BaseAttribute):
    __config_items__ = {"init_mean": [float, None],
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

        if ('gauss' in init_type) or (init_type == 'normal'):
            return self.clamp(gauss(mean, stdev), config)

        if init_type == 'uniform':
            min_value = max(getattr(config, self.min_value_name),
                            (mean-(2*stdev)))
            max_value = min(getattr(config, self.max_value_name),
                            (mean+(2*stdev)))
            return uniform(min_value, max_value)

        raise RuntimeError("Unknown init_type {!r} for {!s}".format(getattr(config, self.init_type_name), self.init_type_name))

    def mutate_value(self, value, config):
         # mutate_rate is usually no lower than replace_rate, and frequently higher - so put first for efficiency
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
        pass


class BoolAttribute(BaseAttribute):
    __config_items__ = {"default": [bool, None],
                        "mutate_rate": [float, None]}

    def init_value(self, config):
        default = getattr(config, self.default_name)

        if default is None:
            return random() < 0.5

        return default

    def mutate_value(self, value, config):
        mutate_rate = getattr(config, self.mutate_rate_name)

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
        pass


class StringAttribute(BaseAttribute):
    __config_items__ = {"default": [str, 'random'],
                        "options": [list, None],
                        "mutate_rate": [float, None]}

    def init_value(self, config):
        default = getattr(config, self.default_name)

        if default in (None, 'random'):
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
        pass
