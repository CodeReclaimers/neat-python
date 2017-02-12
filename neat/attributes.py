from random import choice, gauss, random
from neat.config import ConfigParameter

# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.


class BaseAttribute(object):
    def __init__(self, name):
        self.name = name
        for n, cname in zip(self.__config_items__, self.config_item_names()):
            setattr(self, n + "_name", cname)

    def config_item_names(self):
        return ["{0}_{1}".format(self.name, i) for i in self.__config_items__]


class FloatAttribute(BaseAttribute):
    __config_items__ = ["init_mean",
                        "init_stdev",
                        "replace_rate",
                        "mutate_rate",
                        "mutate_power",
                        "max_value",
                        "min_value"]

    def get_config_params(self):
        return [ConfigParameter(n, float) for n in self.config_item_names()]

    def clamp(self, value, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        return max(min(value, max_value), min_value)

    def init_value(self, config):
        mean = getattr(config, self.init_mean_name)
        stdev = getattr(config, self.init_stdev_name)
        return self.clamp(gauss(mean, stdev), config)

    def mutate_value(self, value, config):
        replace_rate = getattr(config, self.replace_rate_name)

        r = random()
        if r < replace_rate:
            return self.init_value(config)

        mutate_rate = getattr(config, self.mutate_rate_name)
        if r < replace_rate + mutate_rate:
            mutate_power = getattr(config, self.mutate_power_name)
            return value + gauss(0.0, mutate_power)

        return self.clamp(value, config)

    def validate(self, config):
        pass


class BoolAttribute(BaseAttribute):
    __config_items__ = ["default",
                        "mutate_rate"]

    def get_config_params(self):
        default_name, rate_name = self.config_item_names()
        return [ConfigParameter(default_name, bool), ConfigParameter(rate_name, float)]

    def init_value(self, config):
        default = getattr(config, self.default_name)

        if default is None:
            return random() < 0.5

        return default

    def mutate_value(self, value, config):
        mutate_rate = getattr(config, self.mutate_rate_name)

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
    __config_items__ = ["default",
                        "options",
                        "mutate_rate"]

    def get_config_params(self):
        default_name, opt_name, rate_name = self.config_item_names()
        return [ConfigParameter(default_name, str),
                ConfigParameter(opt_name, list),
                ConfigParameter(rate_name, float)]

    def init_value(self, config):
        default = getattr(config, self.default_name)

        if default is None:
            options = getattr(config, self.options_name)
            return choice(options)

        return default

    def mutate_value(self, value, config):
        mutate_rate = getattr(config, self.mutate_rate_name)

        r = random()
        if r < mutate_rate:
            options = getattr(config, self.options_name)
            return choice(options)

        return value

    def validate(self, config):
        pass