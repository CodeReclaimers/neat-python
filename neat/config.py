"""Does general configuration parsing; used by other classes for their configuration."""

from __future__ import annotations

import os
import warnings
from configparser import ConfigParser
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Type


@dataclass(eq=False)
class ConfigParameter:
    """Contains information about one configuration item.

    This is an internal helper used by NEAT's configuration system. It is kept
    intentionally small and value-like, but equality is left as identity to
    avoid changing any external expectations.
    """

    name: str
    value_type: Type[Any]
    default: Any = None
    optional: bool = False  # If True, parameter can be omitted from config

    def __repr__(self) -> str:
        if self.default is None:
            return f"ConfigParameter({self.name!r}, {self.value_type!r})"
        return f"ConfigParameter({self.name!r}, {self.value_type!r}, {self.default!r})"

    def parse(self, section: str, config_parser: ConfigParser) -> Any:
        """Parse a value from *section* of the given ``ConfigParser``.

        Any missing required parameter results in an exception that is handled
        by :class:`Config` to provide a user-friendly error message.
        """

        # Check if parameter exists in config
        if not config_parser.has_option(section, self.name):
            if self.optional:
                return self.default
            # Will raise exception in Config.__init__ exception handler
            raise Exception(f"Missing parameter: {self.name}")

        if self.value_type is int:
            return config_parser.getint(section, self.name)
        if self.value_type is bool:
            return config_parser.getboolean(section, self.name)
        if self.value_type is float:
            return config_parser.getfloat(section, self.name)
        if self.value_type is list:
            v = config_parser.get(section, self.name)
            return v.split(" ")
        if self.value_type is str:
            return config_parser.get(section, self.name)

        raise RuntimeError(f"Unexpected configuration type: {self.value_type!r}")

    def interpret(self, config_dict: Mapping[str, Any], section_name: Optional[str] = None) -> Any:
        """
        Converts the config_parser output into the proper type,
        and checks for errors. All configuration parameters must be explicitly
        specified - no defaults are used.
        
        Args:
            config_dict: Dictionary of configuration values
            section_name: Optional name of the config section (for better error messages)
        """
        value = config_dict.get(self.name)
        if value is None:
            if self.default is None:
                if section_name:
                    raise RuntimeError(f"Missing required configuration item in [{section_name}] section: '{self.name}'")
                else:
                    raise RuntimeError(f"Missing required configuration item: '{self.name}'")
            else:
                # For v1.0: Require all parameters to be explicitly specified
                section_info = f" in [{section_name}] section" if section_name else ""
                raise RuntimeError(
                    f"Missing required configuration item{section_info}: '{self.name}'\n"
                    f"This parameter must be explicitly specified in your configuration file.\n"
                    f"Suggested value: {self.name} = {self.default}"
                )

        try:
            if str == self.value_type:
                return str(value)
            if int == self.value_type:
                return int(value)
            if bool == self.value_type:
                if value.lower() == "true":
                    return True
                elif value.lower() == "false":
                    return False
                else:
                    raise RuntimeError(self.name + " must be True or False")
            if float == self.value_type:
                return float(value)
            if list == self.value_type:
                return value.split(" ")
        except Exception:
            raise RuntimeError(
                f"Error interpreting config item '{self.name}' with value {value!r} and type {self.value_type}")

        raise RuntimeError("Unexpected configuration type: " + repr(self.value_type))

    def format(self, value: Any) -> str:
        if self.value_type is list:
            return " ".join(value)
        return str(value)


def write_pretty_params(f, config: Any, params: Iterable[ConfigParameter]) -> None:
    param_names = [p.name for p in params]
    longest_name = max(len(name) for name in param_names)
    param_names.sort()
    param_lookup: Dict[str, ConfigParameter] = {p.name: p for p in params}

    for name in param_names:
        p = param_lookup[name]
        f.write(f'{p.name.ljust(longest_name)} = {p.format(getattr(config, p.name))}\n')


class UnknownConfigItemError(NameError):
    """Error for unknown configuration option - partially to catch typos."""
    pass


class DefaultClassConfig:
    """
    Replaces at least some boilerplate configuration code
    for reproduction, species_set, and stagnation classes.
    """

    def __init__(self, param_dict, param_list, section_name=None):
        self._params = param_list
        self._section_name = section_name
        param_list_names = []
        for p in param_list:
            setattr(self, p.name, p.interpret(param_dict, section_name))
            param_list_names.append(p.name)
        unknown_list = [x for x in param_dict if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown configuration items:\n" +
                                             "\n\t".join(unknown_list))
            raise UnknownConfigItemError(f"Unknown configuration item {unknown_list[0]!s}")

    @classmethod
    def write_config(cls, f, config):
        # pylint: disable=protected-access
        write_pretty_params(f, config, config._params)


class Config:
    """A container for user-configurable parameters of NEAT."""

    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('fitness_criterion', str),
                ConfigParameter('fitness_threshold', float),
                ConfigParameter('reset_on_extinction', bool),
                ConfigParameter('no_fitness_termination', bool, False),
                ConfigParameter('seed', int, None, optional=True)]

    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, filename, config_information=None):
        # Check that the provided types have the required methods.
        assert hasattr(genome_type, 'parse_config')
        assert hasattr(reproduction_type, 'parse_config')
        assert hasattr(species_set_type, 'parse_config')
        assert hasattr(stagnation_type, 'parse_config')

        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type
        self.config_information = config_information

        if not os.path.isfile(filename):
            raise Exception('No such config file: ' + os.path.abspath(filename))

        parameters = ConfigParser()
        with open(filename) as f:
            parameters.read_file(f)

        # NEAT configuration
        if not parameters.has_section('NEAT'):
            raise RuntimeError("'NEAT' section not found in NEAT configuration file.")

        param_list_names = []
        for p in self.__params:
            try:
                setattr(self, p.name, p.parse('NEAT', parameters))
            except Exception as e:
                if p.optional:
                    # Optional parameter - use default if not present
                    setattr(self, p.name, p.default)
                elif p.default is None:
                    # No default available, re-raise the error
                    raise
                else:
                    # For v1.0: Require all parameters to be explicitly specified
                    raise RuntimeError(
                        f"Missing required configuration item in [NEAT] section: '{p.name}'\n"
                        f"This parameter must be explicitly specified in your configuration file.\n"
                        f"Suggested value: {p.name} = {p.default}"
                    ) from e
            param_list_names.append(p.name)
        param_dict = dict(parameters.items('NEAT'))
        unknown_list = [x for x in param_dict if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown (section 'NEAT') configuration items:\n" + "\n\t".join(unknown_list))
            raise UnknownConfigItemError(f"Unknown (section 'NEAT') configuration item {unknown_list[0]!s}")

        # Parse type sections.
        genome_dict = dict(parameters.items(genome_type.__name__))
        self.genome_config = genome_type.parse_config(genome_dict)

        species_set_dict = dict(parameters.items(species_set_type.__name__))
        self.species_set_config = species_set_type.parse_config(species_set_dict)

        stagnation_dict = dict(parameters.items(stagnation_type.__name__))
        self.stagnation_config = stagnation_type.parse_config(stagnation_dict)

        reproduction_dict = dict(parameters.items(reproduction_type.__name__))
        self.reproduction_config = reproduction_type.parse_config(reproduction_dict)

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write('# The `NEAT` section specifies parameters particular to the NEAT algorithm\n')
            f.write('# or the experiment itself.  This is the only required section.\n')
            f.write('[NEAT]\n')
            write_pretty_params(f, self, self.__params)

            f.write(f'\n[{self.genome_type.__name__}]\n')
            self.genome_type.write_config(f, self.genome_config)

            f.write(f'\n[{self.species_set_type.__name__}]\n')
            self.species_set_type.write_config(f, self.species_set_config)

            f.write(f'\n[{self.stagnation_type.__name__}]\n')
            self.stagnation_type.write_config(f, self.stagnation_config)

            f.write(f'\n[{self.reproduction_type.__name__}]\n')
            self.reproduction_type.write_config(f, self.reproduction_config)
