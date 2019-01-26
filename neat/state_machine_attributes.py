from random import random, randrange

from neat.attributes import FloatAttribute, BaseAttribute
from neat.state_machine_network import Condition


class AttributedAttribute(BaseAttribute):
    """
    Class which describes an attribute, but has additional attributes itself.
    """

    def __init__(self, name, added_attribute, **default_dict):
        super().__init__(name, **default_dict)
        self.additional_attribute = added_attribute

    def get_config_params(self):
        # Override get_config_parameters to also get parameters of condition element.
        parameters = super().get_config_params()
        parameters.extend(self.additional_attribute.get_config_params())
        return parameters

    def get_attr(self):
        """" Returns the requested attribute. """
        return self.additional_attribute


class BiasesAttribute(AttributedAttribute):
    _config_items = {}

    def __init__(self, name, **default_dict):
        super().__init__(name, FloatAttribute('bias'), **default_dict)

    def init_value(self, config):
        # Row with a bias for each output value.
        return [self.get_attr().init_value(config) for _ in range(config.num_outputs)]

    def mutate_value(self, value, config):
        return [self.get_attr().mutate_value(i, config) for i in value]


class WeightsAttribute(AttributedAttribute):
    _config_items = {}

    def __init__(self, name, **default_dict):
        super().__init__(name, FloatAttribute('weight'), **default_dict)

    def init_value(self, config):
        return [[self.get_attr().init_value(config) for _ in range(config.num_inputs)]
                for _ in range(config.num_outputs)]

    def mutate_value(self, value, config):
        """ Mutates all the weight as described in the config files.
        This is found to be faster than a tuple variant, with 1.89s vs 2.13s for 100000 trials.
        """
        return [[self.get_attr().mutate_value(i, config) for i in weigh_row] for weigh_row in value]


class ConditionAttribute(AttributedAttribute):
    """This class represents the attribute for one object."""

    _config_items = {"mutate_input_prob": [float, None],
                     "mutate_comp_prob": [float, None]
                     }

    def __init__(self, name, **default_dict):
        super().__init__(name, FloatAttribute('condition_comparator'), **default_dict)

    def init_value(self, config):
        input_sensor = randrange(0, config.num_inputs)
        comparator = Condition.random_operator()
        comparison = self.get_attr().init_value(config)
        return input_sensor, comparator, comparison

    def mutate_value(self, value, config):
        """ Mutates a the values of a condition parameters are:
        value: a list with the following form [sensor_id, operator, comparison]
        config: a config object containing the right variables.
        """
        # Tuple implementation of value appears to be slightly faster than list implementation.

        # Mutate the input used by this condition.
        mutate_rate = getattr(config, self.mutate_input_prob_name)
        r = random()
        new_sensor_id = value[0]
        if r < mutate_rate:
            new_sensor_id = randrange(0, config.num_inputs)

        # Mutate the comparision operator.
        mutate_rate = getattr(config, self.mutate_comp_prob_name)
        new_comparator = value[1]
        r = random()
        if r < mutate_rate:
            new_comparator = Condition.random_operator()

        new_comparison = self.get_attr().mutate_value(value[2], config)
        return new_sensor_id, new_comparator, new_comparison

    def validate(self, config):  # pragma: no cover
        pass


class ConditionsAttribute(AttributedAttribute):
    """ Class for state machines in the state machine evolving scenario. """
    _config_items = {"add_condition_prob": [float, None],
                     "remove_condition_prob": [float, None]}

    def __init__(self, name, **default_dict):
        super().__init__(name, ConditionAttribute('condition'), **default_dict)

    def init_value(self, config):
        return [self.get_attr().init_value(config)]

    def mutate_value(self, value, config):

        removal_rate = getattr(config, self.remove_condition_prob_name)
        new_conditions = []

        for condition in value:
            if random() > removal_rate:  # If should not be removed and being mutated and added to the new conditions.
                new_conditions.append(self.get_attr().mutate_value(condition, config))

        # Add a new condition if this is selected.
        mutate_rate = getattr(config, self.add_condition_prob_name)
        if mutate_rate > 0 and random() < mutate_rate:
            new_conditions.append(self.get_attr().init_value(config))

        return new_conditions

    def validate(self, config):  # pragma: no cover
        pass
