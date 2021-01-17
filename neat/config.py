"""Does general configuration parsing; used by other classes for their configuration."""
from __future__ import print_function, annotations

import os
import warnings
from typing import Dict, List, Optional, Any, Type, Union, TYPE_CHECKING
from configparser import ConfigParser

if TYPE_CHECKING:
    from neat.genome import DefaultGenome, DefaultGenomeConfig
    from neat.reproduction import DefaultReproduction
    from neat.species import DefaultSpeciesSet
    from neat.stagnation import DefaultStagnation

ConfigValueType = Union[Type[str], Type[int], Type[float], Type[bool], Type[list]]
ConfigInstanceType = Union[str, int, float, bool, list]


class ConfigParameter(object):
    """Contains information about one configuration item.

    一つの設定アイテムの名前タイプ情報を持つ。
    **実際の値はもたない。**
    parseメソッドで、値を持つconfig_parserから該当するself.nameの値を返している（そのため、Parameter自体は値を保持しない　タイプ情報と名前のみ）
    """

    def __init__(self, name: str, value_type: ConfigValueType, default=None):
        self.name: str = name
        self.value_type: ConfigValueType = value_type
        self.default = default

    def __repr__(self):
        if self.default is None:
            return "ConfigParameter({!r}, {!r})".format(self.name,
                                                        self.value_type)
        return "ConfigParameter({!r}, {!r}, {!r})".format(self.name,
                                                          self.value_type,
                                                          self.default)

    def parse(self, section: str, config_parser: ConfigParser) -> ConfigInstanceType:
        """ある一つの情報をパースする.

        sectionは設定ファイルの[NEAT], [DefaultGeneme]
        などのNEATの部分（おそらく）
        なんかsection = NEATしか入ってない
        """
        if int == self.value_type:
            return config_parser.getint(section, self.name)
        if bool == self.value_type:
            return config_parser.getboolean(section, self.name)
        if float == self.value_type:
            return config_parser.getfloat(section, self.name)
        if list == self.value_type:
            v = config_parser.get(section, self.name)
            return v.split(" ")
        if str == self.value_type:
            return config_parser.get(section, self.name)

        raise RuntimeError("Unexpected configuration type: " + repr(self.value_type))

    def interpret(self, config_dict: Dict[str, str]) -> ConfigInstanceType:
        """
        Converts the config_parser output into the proper type,
        supplies defaults if available and needed, and checks for some errors.

        正しい形で一つの設定アイテム情報を返却する
        デフォルト値も設定する
        Genomeクラスで使われている　（ConfigParametaの型とconfig_dictが与えられて　Parameterの型にキャストした実際の値を返す）
        """
        value: str = config_dict.get(self.name)
        if value is None:
            if self.default is None:
                raise RuntimeError('Missing configuration item: ' + self.name)
            else:
                warnings.warn("Using default {!r} for '{!s}'".format(self.default, self.name),
                              DeprecationWarning)
                if (str != self.value_type) and isinstance(self.default, self.value_type):
                    return self.default
                else:
                    value = self.default

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
            raise RuntimeError("Error interpreting config item '{}' with value {!r} and type {}".format(
                self.name, value, self.value_type))

        raise RuntimeError("Unexpected configuration type: " + repr(self.value_type))

    def format(self, value):
        if list == self.value_type:
            return " ".join(value)
        return str(value)


def write_pretty_params(f, config, params):
    param_names = [p.name for p in params]
    longest_name = max(len(name) for name in param_names)
    param_names.sort()
    params = dict((p.name, p) for p in params)

    for name in param_names:
        p = params[name]
        f.write('{} = {}\n'.format(p.name.ljust(longest_name), p.format(getattr(config, p.name))))


class UnknownConfigItemError(NameError):
    """Error for unknown configuration option - partially to catch typos."""
    pass


class DefaultClassConfig(object):
    """
    Replaces at least some boilerplate configuration code
    for reproduction, species_set, and stagnation classes.

    - compatibility-threshold
    - species-fitness-func, max-stagnation
    - elitism, survival_threashold

    のようにDefaultSpeciesSet, DefaultStagnation, DefaultReproductionが**それぞれ**でparam_dict, param_listで与えられる
    種類ごとにデフォルトクラスコンフィグが作られる
    おそらくSpeciesSetのように特定クラスのために設定値をいれるもの？
    """

    def __init__(self, param_dict: Dict[str, str], param_list: List[ConfigParameter]) -> None:
        self._params: List[ConfigParameter] = param_list
        param_list_names: List[str] = []
        for p in param_list:
            setattr(self, p.name, p.interpret(param_dict))
            param_list_names.append(p.name)
        unknown_list = [x for x in param_dict if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown configuration items:\n" +
                                             "\n\t".join(unknown_list))
            raise UnknownConfigItemError("Unknown configuration item {!s}".format(unknown_list[0]))

    @classmethod
    def write_config(cls, f, config) -> None:
        # pylint: disable=protected-access
        write_pretty_params(f, config, config._params)


class Config(object):
    """A simple container for user-configurable parameters of NEAT.

    設定ファイルの`NEAT`のみの値を入れるものっぽい
    と思ったけどすべての項目がConfigに入る
    """

    __params: List[ConfigParameter] = [ConfigParameter('pop_size', int),
                                       ConfigParameter('fitness_criterion', str),
                                       ConfigParameter('fitness_threshold', float),
                                       ConfigParameter('reset_on_extinction', bool),
                                       ConfigParameter('no_fitness_termination', bool, False)]

    def __init__(self, genome_type: Type[DefaultGenome], reproduction_type: Type[DefaultReproduction], species_set_type: Type[DefaultSpeciesSet], stagnation_type: Type[DefaultStagnation], filename: str) -> None:

        # members
        self.fitness_criterion: str = ""
        self.fitness_threshold: float = 0
        self.pop_size: int = 0
        self.reset_on_extinction: bool = False
        self.no_fitness_termination: bool = False

        # Check that the provided types have the required methods.
        assert hasattr(genome_type, 'parse_config')
        assert hasattr(reproduction_type, 'parse_config')
        assert hasattr(species_set_type, 'parse_config')
        assert hasattr(stagnation_type, 'parse_config')

        self.genome_type: Type[DefaultGenome] = genome_type
        self.reproduction_type: Type[DefaultReproduction] = reproduction_type
        self.species_set_type: Type[DefaultSpeciesSet] = species_set_type
        self.stagnation_type: Type[DefaultStagnation] = stagnation_type

        if not os.path.isfile(filename):
            raise Exception('No such config file: ' + os.path.abspath(filename))

        parameters: ConfigParser = ConfigParser()
        with open(filename) as f:
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)

        # NEAT configuration
        if not parameters.has_section('NEAT'):
            raise RuntimeError("'NEAT' section not found in NEAT configuration file.")

        param_list_names: List[str] = []
        for p in self.__params:
            if p.default is None:
                setattr(self, p.name, p.parse('NEAT', parameters))
            else:
                try:
                    setattr(self, p.name, p.parse('NEAT', parameters))
                except Exception:
                    setattr(self, p.name, p.default)
                    warnings.warn("Using default {!r} for '{!s}'".format(p.default, p.name),
                                  DeprecationWarning)
            param_list_names.append(p.name)
        param_dict: Dict[str, str] = dict(parameters.items('NEAT'))
        unknown_list = [x for x in param_dict if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown (section 'NEAT') configuration items:\n" +
                                             "\n\t".join(unknown_list))
            raise UnknownConfigItemError(
                "Unknown (section 'NEAT') configuration item {!s}".format(unknown_list[0]))

        # Parse type sections.
        # genome_type=DefaultGenomeに関する設定値をgenome_dictにDictとして入れる
        # そしてParseしてint, strのようなそれぞれの値にしている
        # GenomeだけはDefaultGenomeConfig　それ以外はDefaultClassConfig
        genome_dict: Dict[str, str] = dict(parameters.items(genome_type.__name__))
        self.genome_config: DefaultGenomeConfig = genome_type.parse_config(genome_dict)

        species_set_dict: Dict[str, str] = dict(parameters.items(species_set_type.__name__))
        self.species_set_config: DefaultClassConfig = species_set_type.parse_config(species_set_dict)

        stagnation_dict: Dict[str, str] = dict(parameters.items(stagnation_type.__name__))
        self.stagnation_config: DefaultClassConfig = stagnation_type.parse_config(stagnation_dict)

        reproduction_dict: Dict[str, str] = dict(parameters.items(reproduction_type.__name__))
        self.reproduction_config: DefaultClassConfig = reproduction_type.parse_config(reproduction_dict)

    def save(self, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write('# The `NEAT` section specifies parameters particular to the NEAT algorithm\n')
            f.write('# or the experiment itself.  This is the only required section.\n')
            f.write('[NEAT]\n')
            write_pretty_params(f, self, self.__params)

            f.write('\n[{0}]\n'.format(self.genome_type.__name__))
            self.genome_type.write_config(f, self.genome_config)

            f.write('\n[{0}]\n'.format(self.species_set_type.__name__))
            self.species_set_type.write_config(f, self.species_set_config)

            f.write('\n[{0}]\n'.format(self.stagnation_type.__name__))
            self.stagnation_type.write_config(f, self.stagnation_config)

            f.write('\n[{0}]\n'.format(self.reproduction_type.__name__))
            self.reproduction_type.write_config(f, self.reproduction_config)
