
Module summaries
==================

.. default-role:: any

.. index:: ! activation function

.. py:module:: activations
   :synopsis: Has the built-in activation functions, code for using them,  and code for adding new user-defined ones.

activations
---------------
Has the built-in :term:`activation functions <activation function>`, code for using them, and code for adding new user-defined ones.

  .. py:exception:: InvalidActivationFunction(TypeError)

    Exception called if an activation function being added is invalid according to the `validate_activation` function, or if an unknown activation
    function is requested by name via :py:meth:`get <ActivationFunctionSet.get()>`.

    .. versionchanged:: 0.92
      Base of exception changed to more-precise TypeError.

  .. py:function:: validate_activation(function)

    Checks to make sure its parameter is a function that takes a single argument.

    :param function: Object to be checked.
    :type function: :datamodel:`object <objects-values-and-types>`
    :raises InvalidActivationFunction: If the object does not pass the tests.

  .. py:class:: ActivationFunctionSet

    Contains the list of current valid activation functions, including methods for adding and getting them.

    .. py:method:: add(name, function)

      After validating the function (via `validate_activation`), adds it to the available activation functions under the given name. Used
      by :py:meth:`DefaultGenomeConfig.add_activation <genome.DefaultGenomeConfig.add_activation>`.

      :param str name: The name by which the function is to be known in the :ref:`configuration file <activation-function-config-label>`.
      :param function: The function to be added.
      :type function: `function`

    .. py:method:: get(name)

      Returns the named function, or raises an exception if it is not a known activation function.

      :param str name: The name of the function.
      :return: The function of interest
      :rtype: `function`
      :raises InvalidActivationFunction: If the function is not known.

    .. py:method:: is_valid(name)

      Checks whether the named function is a known activation function.

      :param str name: The name of the function.
      :return: Whether or not the function is known.
      :rtype: :pytypes:`bool <typesnumeric>`

.. index:: ! aggregation function

.. py:module:: aggregations
   :synopsis: Has the built-in aggregation functions, code for using them,  and code for adding new user-defined ones.

aggregations
---------------
Has the built-in :term:`aggregation functions <aggregation function>`, code for using them, and code for adding new user-defined ones.

  .. note::

    :term:`Non-enabled <enabled>` :term:`connections <connection>` will, by all methods currently included in NEAT-Python, *not* be included among
    the numbers input to these functions, even as 0s.

  .. py:function:: product_aggregation(x)

    An adaptation of the multiplication function to take an :pygloss:`iterable`.

    :param x: The numbers to be multiplied together; takes any ``iterable``.
    :type x: list(:pytypes:`float <typesnumeric>`) or tuple(:pytypes:`float <typesnumeric>`) or set(:pytypes:`float <typesnumeric>`)
    :return: :math:`\prod(x)`
    :rtype: :pytypes:`float <typesnumeric>`

  .. py:function:: sum_aggregation(x)

    Probably the most commonly-used aggregation function.

    :param x: The numbers to find the sum of; takes any :pygloss:`iterable`.
    :type x: list(:pytypes:`float <typesnumeric>`) or tuple(:pytypes:`float <typesnumeric>`) or set(:pytypes:`float <typesnumeric>`)
    :return: :math:`\sum(x)`
    :rtype: :pytypes:`float <typesnumeric>`

  .. py:function:: max_aggregation(x)

    Returns the maximum of the inputs.

    :param x: The numbers to find the greatest of; takes any :pygloss:`iterable`.
    :type x: list(:pytypes:`float <typesnumeric>`) or tuple(:pytypes:`float <typesnumeric>`) or set(:pytypes:`float <typesnumeric>`)
    :return: :math:`\max(x)`
    :rtype: :pytypes:`float <typesnumeric>`

  .. py:function:: min_aggregation(x)

    Returns the minimum of the inputs.

    :param x: The numbers to find the least of; takes any :pygloss:`iterable`.
    :type x: list(:pytypes:`float <typesnumeric>`) or tuple(:pytypes:`float <typesnumeric>`) or set(:pytypes:`float <typesnumeric>`)
    :return: :math:`\min(x)`
    :rtype: :pytypes:`float <typesnumeric>`

  .. py:function:: maxabs_aggregation(x)

    Returns the maximum by absolute value, which may be positive or negative. Envisioned as suitable for neural network pooling operations.

    :param x: The numbers to find the absolute-value maximum of; takes any :pygloss:`iterable`.
    :type x: list(:pytypes:`float <typesnumeric>`) or tuple(:pytypes:`float <typesnumeric>`) or set(:pytypes:`float <typesnumeric>`)
    :return: :math:`x_i, i = \text{argmax}\lvert\mathbf{x}\rvert`
    :rtype: :pytypes:`float <typesnumeric>`

    .. versionadded:: 0.92

  .. py:function:: median_aggregation(x)

    Returns the :py:func:`median <math_util.median2>` of the inputs.

    :param x: The numbers to find the median of; takes any :pygloss:`iterable`.
    :type x: list(:pytypes:`float <typesnumeric>`) or tuple(:pytypes:`float <typesnumeric>`) or set(:pytypes:`float <typesnumeric>`)
    :return: The median; if there are an even number of inputs, takes the mean of the middle two.
    :rtype: :pytypes:`float <typesnumeric>`

    .. versionadded:: 0.92

  .. py:function:: mean_aggregation(x)

    Returns the arithmetic mean. Potentially maintains a more stable result than ``sum`` for changing numbers of :term:`enabled`
    :term:`connections <connection>`, which may be good or bad depending on the circumstances; having both available to the algorithm is advised.

    :param x: The numbers to find the mean of; takes any :pygloss:`iterable`.
    :type x: list(:pytypes:`float <typesnumeric>`) or tuple(:pytypes:`float <typesnumeric>`) or set(:pytypes:`float <typesnumeric>`)
    :return: The arithmetic mean.
    :rtype: :pytypes:`float <typesnumeric>`

    .. versionadded:: 0.92

  .. py:exception:: InvalidAggregationFunction(TypeError)

    Exception called if an aggregation function being added is invalid according to the `validate_aggregation` function, or if an unknown aggregation
    function is requested by name via :py:meth:`get <AggregationFunctionSet.get()>`.

    .. versionadded:: 0.92

  .. py:function:: validate_aggregation(function)

    Checks to make sure its parameter is a function that takes at least one argument.

    :param function: Object to be checked.
    :type function: :datamodel:`object <objects-values-and-types>`
    :raises InvalidAggregationFunction: If the object does not pass the tests.

    .. versionadded:: 0.92

  .. py:class:: AggregationFunctionSet

    Contains the list of current valid aggregation functions, including methods for adding and getting them.

    .. py:method:: add(name, function)

      After validating the function (via `validate_aggregation`), adds it to the available activation functions under the given name. Used
      by :py:meth:`DefaultGenomeConfig.add_activation <genome.DefaultGenomeConfig.add_activation>`. TODO: Check for whether
      the function needs `reduce <functools.reduce>`, or at least offer a form of this function (or extra argument for it, defaulting to false)
      and/or its interface in :py:mod:`genome`, that will appropriately "wrap" the input function.

      :param str name: The name by which the function is to be known in the :ref:`configuration file <aggregation-function-config-label>`.
      :param function: The function to be added.
      :type function: `function`

      .. versionadded:: 0.92

    .. py:method:: get(name)

      Returns the named function, or raises an exception if it is not a known aggregation function.

      :param str name: The name of the function.
      :return: The function of interest
      :rtype: `function`
      :raises InvalidAggregationFunction: If the function is not known.

      .. versionadded:: 0.92

    .. py:method:: __getitem__(index)

      Present for compatibility with older programs that expect the aggregation functions to be in a `dict <dictionary>`. A wrapper for
      :py:meth:`get(index) <AggregationFunctionSet.get()>`.

      :param str index: The name of the function.
      :return: The function of interest.
      :rtype: `function`
      :raises InvalidAggregationFunction: If the function is not known.
      :raises DeprecationWarning: Always.

      .. versionchanged:: 0.92
        Originally a dictionary in :py:mod:`genome`.

      .. deprecated:: 0.92
        Use :py:meth:`get(index) <AggregationFunctionSet.get()>` instead.

    .. py:method:: is_valid(name)

      Checks whether the named function is a known aggregation function.

      :param str name: The name of the function.
      :return: Whether or not the function is known.
      :rtype: :pytypes:`bool <typesnumeric>`

      .. versionadded:: 0.92

  .. versionchanged:: 0.92
    Moved from :py:mod:`genome` and expanded to match `activations` (plus the ``maxabs``, ``median``, and ``mean`` functions added).

.. py:module:: attributes
   :synopsis: Deals with attributes used by genes.

attributes
-------------
Deals with :term:`attributes` used by :term:`genes <gene>`.

  .. inheritance-diagram:: attributes

  .. py:class:: BaseAttribute(name, **default_dict)

    Superclass for the type-specialized attribute subclasses, used by genes (such as via the :py:class:`genes.BaseGene` implementation). Updates
    ``_config_items`` with any defaults supplied, then uses `config_item_name` to set up a listing of the names of configuration items using `setattr`.

    :param str name: The name of the attribute, held in the instance's ``name`` attribute.
    :param default_dict: An optional dictionary of defaults for the configuration items.
    :type default_dict: dict(str, str)

    .. versionchanged:: 0.92
      Default_dict capability added.

    .. py:method:: config_item_name(config_item_base_name)

      Formats a configuration item's name by combining the attribute's name with the base item name.

      :param str config_item_base_name: The base name of the configuration item, to be combined with the attribute's name.
      :return: The configuration item's full name.
      :rtype: str

      .. versionchanged:: 0.92
        Originally (as ``config_item_names``) did not take any input and returned a list based on the ``_config_items`` subclass attribute.

    .. py:method:: get_config_params()

      Uses `config_item_name` for each configuration item to get the name, then gets the appropriate type of :py:class:`config.ConfigParameter`
      instance for each (with any appropriate defaults being set from ``_config_items``, including as modified by `BaseAttribute`) and returns it.

      :return: A list of ``ConfigParameter`` instances.
      :rtype: list(:datamodel:`instance <index-48>`)

      .. versionchanged:: 0.92
        Was originally specific for the attribute subclass, since it did not pick up the appropriate type from the ``_config_items`` list; default capability
        also added.

  .. py:class:: FloatAttribute(BaseAttribute)

    Class for numeric :term:`attributes` such as the :term:`response` of a :term:`node`; includes code for configuration, creation, and mutation.

    .. index:: ! max_value
    .. index:: ! min_value

    .. py:method:: clamp(value, config)

      Gets the minimum and maximum values desired from ``config``, then ensures that the value is between them.

      :param value: The value to be clamped.
      :type value: :pytypes:`float <typesnumeric>`
      :param config: The configuration object from which the minimum and maximum desired values are to be retrieved.
      :type config: :datamodel:`instance <index-48>`
      :return: The value, if it is within the desired range, or the appropriate end of the range, if it is not.
      :rtype: :pytypes:`float <typesnumeric>`

    .. index:: init_mean
    .. index:: init_stdev
    .. index:: init_type

    .. py:method:: init_value(config)

      Initializes the attribute's value, using either a gaussian distribution with the configured mean and standard deviation, followed by `clamp` to
      keep the result within the desired range, or a uniform distribution, depending on the configuration setting of ``init_type``.

      :param config: The configuration object from which the mean, standard deviation, and initialization distribution type values are to be retrieved.
      :type config: :datamodel:`instance <index-48>`
      :return: The new value.
      :rtype: :pytypes:`float <typesnumeric>`

      .. versionchanged:: 0.92
        Uniform distribution initialization option added.

    .. index:: ! mutation
    .. index:: ! mutate_power
    .. index:: ! replace_rate
    .. index:: mutate_rate

    .. py:method:: mutate_value(value, config)

      May replace (as if reinitializing, using `init_value`), mutate (using a 0-mean gaussian distribution with a configured standard
      deviation from ``mutate_power``), or leave alone the input value, depending on the configuration settings (of ``replace_rate`` and ``mutate_rate``).

      :param value: The current value of the attribute.
      :type value: :pytypes:`float <typesnumeric>`
      :param config: The configuration object from which the parameters are to be extracted.
      :type config: :datamodel:`instance <index-48>`
      :return: Either the original value, if unchanged, or the new value.
      :rtype: :pytypes:`float <typesnumeric>`

  .. py:class:: BoolAttribute(BaseAttribute)

    Class for boolean :term:`attributes` such as whether a :term:`connection` is :term:`enabled` or not; includes code for configuration, creation, and mutation.

    .. index:: ! X_default

    .. py:method:: init_value(config)

      Initializes the attribute's value, either using a configured ``default``, or (if the default is "random") with a 50/50 chance of `True` or `False`.

      .. deprecated:: 0.92
        While it is possible to use "None" as an equivalent to "random", this is too easily confusable with an actual `None`.

      .. versionchanged:: 0.92
        Ability to use "random" for a 50/50 chance of `True` or `False` added.

      :param config: The configuration object from which the default parameter is to be retrieved.
      :type config: :datamodel:`instance <index-48>`
      :return: The new value.
      :rtype: :pytypes:`bool <typesnumeric>`
      :raises RuntimeError: If the default value is not recognized as standing for any of `True`, `False`, "random", or "none".

    .. index:: ! mutation
    .. index:: mutate_rate
    .. index:: ! rate_to_false_add
    .. index:: ! rate_to_true_add

    .. py:method:: mutate_value(value, config)

      With a frequency determined by the ``mutate_rate`` and ``rate_to_false_add`` or
      ``rate_to_true_add`` configuration parameters, replaces the value with a 50/50 chance of ``True`` or ``False``; note that this has a
      50% chance of leaving the value unchanged.

      :param bool value: The current value of the attribute.
      :param config: The configuration object from which the ``mutate_rate`` and other parameters are to be extracted.
      :type config: :datamodel:`instance <index-48>`
      :return: Either the original value, if unchanged, or the new value.
      :rtype: :pytypes:`bool <typesnumeric>`

      .. versionchanged:: 0.92
        Added the ``rate_to_false_add`` and ``rate_to_true_add`` parameters.

  .. py:class:: StringAttribute(BaseAttribute)

    Class for string attributes such as the :term:`aggregation function` of a :term:`node`, which are selected from a list of options;
    includes code for configuration, creation, and mutation.

    .. index:: ! X_default
    .. index:: X_options
    .. index::
      see: default; X_default
      see: options; X_options

    .. py:method:: init_value(config)

      Initializes the attribute's value, either using a configured ``default`` or (if the default is "random") with a
      randomly-chosen member of the ``options`` (each having an equal chance). Note: It is possible for the default value, if specifically configured, to
      **not** be one of the options.

      .. deprecated:: 0.92
        While it is possible to use "None" as an equivalent to "random", this is too easily confusable with an actual `None`.

      :param config: The configuration object from which the default and, if necessary, ``options`` parameters are to be retrieved.
      :type config: :datamodel:`instance <index-48>`
      :return: The new value.
      :rtype: str

    .. index:: ! mutation
    .. index:: mutate_rate
    .. index:: ! X_options

    .. py:method:: mutate_value(value, config)

      With a frequency determined by the ``mutate_rate`` configuration parameter, replaces
      the value with one of the ``options``, with each having an equal chance; note that this can be the same value as before.
      (It is possible to crudely alter the chances of what is chosen by listing a given option more than once, although this is inefficient given the use of the
      `random.choice` function.)
      TODO: Add configurable probabilities of which option is used. Longer-term, as with the
      improved version of RBF-NEAT, separate genes for the likelihoods of each (but always doing some change, to prevent overly-conservative evolution
      due to its inherent short-sightedness), allowing the genomes to control the distribution of options, will be desirable.

      :param str value: The current value of the attribute.
      :param config: The configuration object from which the ``options`` and other parameters are to be extracted.
      :type config: :datamodel:`instance <index-48>`
      :return: The new value.
      :rtype: str

  .. versionchanged:: 0.92
    ``__config_items__`` changed to ``_config_items``, since it is not a Python internal variable.

.. py:module:: checkpoint
   :synopsis: Uses `pickle` to save and restore populations (and other aspects of the simulation state).

checkpoint
---------------
Uses :py:mod:`pickle` to save and restore populations (and other aspects of the simulation state).

  .. note::

    The speed of this module can vary widely between python implementations (and perhaps versions).

  .. py:class:: Checkpointer(generation_interval=100, time_interval_seconds=300, filename_prefix='neat-checkpoint-')

    A reporter class that performs checkpointing, saving and restoring the simulation state (including population, randomization, and other aspects).
    It saves the current state every ``generation_interval`` generations or ``time_interval_seconds`` seconds, whichever happens first.
    Subclasses :py:class:`reporting.BaseReporter`. (The potential save point is at the end of a generation.) The start of the filename will be equal
    to ``filename_prefix``, followed by the generation number. If there is a need to check the last generation for which a checkpoint was saved, such as to
    determine which file to load, access ``last_generation_checkpoint``; if -1, none have been saved.

    :param generation_interval: If not None, maximum number of generations between checkpoints.
    :type generation_interval: :pytypes:`int <typesnumeric>` or None
    :param time_interval_seconds: If not None, maximum number of seconds between checkpoints.
    :type time_interval_seconds: :pytypes:`float <typesnumeric>` or None
    :param str filename_prefix: The prefix for the checkpoint file names.

    .. py:method:: save_checkpoint(config, population, species, generation)

      Saves the current simulation (including randomization) state to (if using the default ``neat-checkpoint-`` for ``filename_prefix``)
      :file:`neat-checkpoint-{generation}`, with ``generation`` being the generation number.

      :param config: The `config.Config` configuration instance to be used.
      :type config: :datamodel:`instance <index-48>`
      :param population: A population as created by :py:meth:`reproduction.DefaultReproduction.create_new` or a compatible implementation.
      :type population: dict(int, :datamodel:`object <objects-values-and-types>`)
      :param species: A :py:class:`species.DefaultSpeciesSet` (or compatible implementation) instance.
      :type species: :datamodel:`instance <index-48>`
      :param generation: The generation number.
      :type generation: :pytypes:`int <typesnumeric>`

    .. py:staticmethod:: restore_checkpoint(filename)

      Resumes the simulation from a previous saved point. Loads the specified file, sets the randomization state, and returns
      a :py:class:`population.Population` object set up with the rest of the previous state.

      :param str filename: The file to be restored from.
      :return: :py:class:`Population <population.Population>` instance that can be used with :py:meth:`Population.run <population.Population.run>` to restart the simulation.
      :rtype:  :datamodel:`instance <index-48>` 

.. index:: fitness_criterion
.. index:: fitness_threshold
.. index:: no_fitness_termination
.. index:: pop_size
.. index:: reset_on_extinction
.. index:: generation

.. py:module:: config
   :synopsis: Does general configuration parsing; used by other classes for their configuration.

config
--------
Does general configuration parsing; used by other classes for their configuration.

  .. py:class:: ConfigParameter(name, value_type, default=None)

    Does initial handling of a particular configuration parameter.

    :param str name: The name of the configuration parameter.
    :param value_type: The type that the configuration parameter should be; must be one of `str`, :pytypes:`int <typesnumeric>`, :pytypes:`bool <typesnumeric>`, :pytypes:`float <typesnumeric>`, or `list`.
    :param default: If given, the default to use for the configuration parameter.
    :type default: str or None

    .. versionchanged:: 0.92
      Default capability added.

    .. py:method:: __repr__()

      Returns a representation of the class suitable for use in code for initialization.

      :return: Representation as for `repr`.
      :rtype: str

    .. py:method:: parse(section, config_parser)

      Uses the supplied configuration parser (either from the :py:class:`configparser.ConfigParser` class, or - for 2.7 - the
      `ConfigParser.SafeConfigParser class <https://docs.python.org/2.7/library/configparser.html#ConfigParser.SafeConfigParser>`_) to gather the
      configuration parameter from the appropriate configuration file :ref:`section <configuration-file-sections-label>`. Parsing varies depending on the type.

      :param str section: The section name, taken from the `__name__` attribute of the class to be configured (or ``NEAT`` for those parameters).
      :param config_parser: The configuration parser to be used.
      :type config_parser: :datamodel:`instance <index-48>`
      :return: The configuration parameter value, in stringified form unless a list.
      :rtype: str or list(str)

    .. py:method:: interpret(config_dict)

      Takes a `dictionary <dict>` of configuration parameters, as output by the configuration parser called in :py:meth:`parse`, and interprets them into the
      proper type, with some error-checking.

      :param config_dict: Configuration parameters as output by the configuration parser.
      :type config_dict: dict(str, str)
      :return: The configuration parameter value
      :rtype: str or :pytypes:`int <typesnumeric>` or :pytypes:`bool <typesnumeric>` or :pytypes:`float <typesnumeric>` or list(str)
      :raises RuntimeError: If there is a problem with the configuration parameter.
      :raises DeprecationWarning: If a default is used.

      .. versionchanged:: 0.92
        Default capability added.

    .. py:method:: format(value)

      Depending on the type of configuration parameter, returns either a space-separated list version, for ``list``  parameters, or the stringified version
      (using `str`), of ``value``.

      :param value: Configuration parameter value to be formatted.
      :type value: str or :pytypes:`int <typesnumeric>` or :pytypes:`bool <typesnumeric>` or :pytypes:`float <typesnumeric>` or list
      :return: String version.
      :rtype: str

  .. py:function:: write_pretty_params(f, config, params)

    Prints configuration parameters, with justification based on the longest configuration parameter name.

    :param f: File object to be written to.
    :type f: :pygloss:`file <file-object>`
    :param config: Configuration object from which parameter values are to be fetched (using `getattr`).
    :type config: :datamodel:`instance <index-48>`
    :param params: List of :py:class:`ConfigParameter` instances giving the names of interest and the types of parameters.
    :type params: list(:datamodel:`instance <index-48>`)

  .. py:exception:: UnknownConfigItemError(NameError)

    Error for unknown configuration option(s) - partially to catch typos. TODO: :py:class:`genome.DefaultGenomeConfig` does not currently check for these.

    .. versionadded:: 0.92

  .. py:class:: DefaultClassConfig(param_dict, param_list)

    Replaces at least some boilerplate configuration code for reproduction, species_set, and stagnation classes.

    :param param_dict: Dictionary of configuration parameters from config file.
    :type param_dict: dict(str, str)
    :param param_list: List of `ConfigParameter` instances; used to know what parameters are of interest to the calling class.
    :type param_list: list(:datamodel:`instance <index-48>`)
    :raises UnknownConfigItemError: If a key in ``param_dict`` is not among the names in ``param_list``.

    .. py:classmethod:: write_config(f, config)

      Required method (inherited by calling classes). Uses :py:func:`write_pretty_params` to output parameters of interest to the calling class.

      :param f: File object to be written to.
      :type f: :pygloss:`file <file-object>`
      :param config: DefaultClassConfig instance.
      :type config: :datamodel:`instance <index-48>`

    .. versionadded:: 0.92

  .. index:: fitness criterion
  .. index:: fitness_threshold
  .. index:: no_fitness_termination
  .. index:: pop_size
  .. index:: reset_on_extinction

  .. py:class:: Config(genome_type, reproduction_type, species_set_type, stagnation_type, filename)

    A simple container for user-configurable parameters of NEAT. The four parameters ending in ``_type`` may be the built-in ones or user-provided objects,
    which must make available the methods ``parse_config`` and ``write_config``, plus others depending on which object it is. (For more information on the
    objects, see below and :ref:`customization-label`.) ``Config`` itself takes care of the :ref:`NEAT parameters <configuration-file-NEAT-section-label>`,
    which are found as some of its attributes. For a description of the configuration file, see :ref:`configuration-file-description-label`. The
    :pytypes:`__name__ <definition.__name__>` attributes of the ``_type`` parameters are used for the titles of the configuration file sections. A Config
    instance's ``genome_config``, ``species_set_config``, ``stagnation_config``, and ``reproduction_config`` attributes hold the configuration objects for the
    respective classes.

    :param genome_type: Specifies the genome class used, such as :py:class:`genome.DefaultGenome` or :py:class:`iznn.IZGenome`. See :ref:`genome-interface-label` for the needed interface.
    :type genome_type: :pygloss:`class`
    :param reproduction_type: Specifies the reproduction class used, such as :py:class:`reproduction.DefaultReproduction`. See :ref:`reproduction-interface-label` for the needed interface.
    :type reproduction_type: :pygloss:`class`
    :param species_set_type: Specifies the species set class used, such as :py:class:`species.DefaultSpeciesSet`.
    :type species_set_type: :pygloss:`class`
    :param stagnation_type: Specifies the stagnation class used, such as :py:class:`stagnation.DefaultStagnation`.
    :type stagnation_type: :pygloss:`class`
    :param str filename: Pathname for configuration file to be opened, read, processed by a parser from the :py:class:`configparser.ConfigParser` class (or, for 2.7, the `ConfigParser.SafeConfigParser class <https://docs.python.org/2.7/library/configparser.html#ConfigParser.SafeConfigParser>`_), the ``NEAT`` section handled by ``Config``, and then other sections passed to the ``parse_config`` methods of the appropriate classes.
    :raises AssertionError: If any of the ``_type`` classes lack a ``parse_config`` method.
    :raises UnknownConfigItemError: If an option in the ``NEAT`` section of the configuration file is not recognized.
    :raises DeprecationWarning: If a default is used for one of the ``NEAT`` section options.

    .. versionchanged:: 0.92
      Added default capabilities, UnknownConfigItemError, no_fitness_termination.

    .. py:method:: save(filename)

      Opens the specified file for writing (not appending) and outputs a configuration file from the current configuration. Uses :py:func:`write_pretty_params` for
      the ``NEAT`` parameters and the appropriate class ``write_config`` methods for the other sections. (A comparison of it and the input configuration file
      can be used to determine any default parameters of interest.)

      :param str filename: The configuration file to be written.

.. py:module:: ctrnn
   :synopsis: Handles the continuous-time recurrent neural network implementation.

ctrnn
-------

  .. py:class:: CTRNNNodeEval(time_constant, activation, aggregation, bias, response, links)

    Sets up the basic :doc:`ctrnn <ctrnn>` (:term:`continuous-time` :term:`recurrent` neural network) :term:`nodes <node>`.

    :param float time_constant: Controls how fast the node responds; :math:`\tau_i` from :doc:`ctrnn`.
    :param activation: :term:`Activation function <activation function>` for the node.
    :type activation: `function`
    :param aggregation: :term:`Aggregation function <aggregation function>` for the node.
    :type aggregation: `function`
    :param bias: :term:`Bias <bias>` for the node.
    :type bias: :pytypes:`float <typesnumeric>`
    :param response: :term:`Response <response>` multiplier for the node.
    :type response: :pytypes:`float <typesnumeric>`
    :param links: List of other nodes providing input, as tuples of (input :term:`key`, :term:`weight`)
    :type links: list(tuple(int,float))

  .. py:class:: CTRNN(inputs, outputs, node_evals)

    Sets up the :doc:`ctrnn <ctrnn>` network itself.

    .. index:: recurrent

    .. py:method:: reset()

      Resets the time and all node activations to 0 (necessary due to otherwise retaining state via :term:`recurrent` connections).

    .. index:: ! continuous-time

    .. py:method:: advance(inputs, advance_time, time_step=None)

      Advance the simulation by the given amount of time, assuming that inputs are
      constant at the given values during the simulated time.

      :param inputs: The values for the :term:`input nodes <input node>`.
      :type inputs: list(float)
      :param advance_time: How much time to advance the network before returning the resulting outputs.
      :type advance_time: :pytypes:`float <typesnumeric>`
      :param time_step: How much time per step to advance the network; the default of ``None`` will currently result in an error, but it is planned to determine it automatically.
      :type time_step: :pytypes:`float <typesnumeric>` or None
      :return: The values for the :term:`output nodes <output node>`.
      :rtype: list(float)
      :raises NotImplementedError: If a ``time_step`` is not given.
      :raises RuntimeError: If the number of ``inputs`` does not match the number of :term:`input nodes <input node>`

      .. versionchanged:: 0.92
        Exception changed to more-specific RuntimeError.

    .. py:staticmethod:: create(genome, config, time_constant)

      Receives a genome and returns its phenotype (a :py:class:`CTRNN` with :py:class:`CTRNNNodeEval` :term:`nodes <node>`).

      :param genome: A :py:class:`genome.DefaultGenome` instance.
      :type genome: :datamodel:`instance <index-48>`
      :param config: A :py:class:`config.Config` instance.
      :type config: :datamodel:`instance <index-48>`
      :param time_constant: Used for the :py:class:`CTRNNNodeEval` initializations.
      :type time_constant: :pytypes:`float <typesnumeric>`


.. index:: ! compute node
.. index:: ! primary node
.. index:: ! secondary node
.. index::
    see: primary compute node; primary node
    see: secondary compute node; secondary node

.. py:module:: distributed
   :synopsis: Distributed evaluation of genomes.

distributed
--------------
  Distributed evaluation of genomes.

  .. note::

    This module is in a **beta** state, and still *unstable* even in single-machine testing. Reliability is likely to vary, including depending on the Python version
    and implementation (e.g., cpython vs pypy) in use and the likelihoods of timeouts (due to machine and/or network slowness). In particular, while the code can try
    to reconnect between between :term:`primary <primary node>` and :term:`secondary <secondary node>` nodes, as noted in the `multiprocessing` documentation
    this may not work due to data loss/corruption. Note also that this module is not responsible for starting the script copies on the different
    :term:`compute nodes <compute node>`, since this is very site/configuration-dependent.

  .. rubric:: About :term:`compute nodes <compute node>`:

  The :term:`primary compute node` (the node which creates and mutates genomes) and the :term:`secondary compute nodes <secondary node>` (the nodes which
  evaluate genomes) can execute the same script. The role of a compute node is determined using the ``mode`` argument of the DistributedEvaluator. If the
  mode is :py:data:`MODE_AUTO`, the `host_is_local()` function is used to check if the ``addr`` argument points to the localhost. If it does, the compute
  node starts as a :term:`primary node`, and otherwise as a :term:`secondary node`. If ``mode`` is :py:data:`MODE_PRIMARY`, the compute node always starts
  as a primary node. If ``mode`` is :py:data:`MODE_SECONDARY`, the compute node will always start as a secondary node.

  There can only be one primary node per NEAT, but any number of secondary nodes. The primary node will not evaluate any genomes, which means you will
  always need at least two compute nodes (one primary and at least one secondary).

  You can run any number of compute nodes on the same physical machine (or VM). However, if a machine has both a primary node and one or more secondary
  nodes, :py:data:`MODE_AUTO` cannot be used for those secondary nodes - :py:data:`MODE_SECONDARY` will need to be specified.

  .. rubric:: Usage:

  1. Import modules and define the evaluation logic (the ``eval_genome`` function). (After this, check for ``if __name__ == '__main__'``, and put the rest of the code inside the body of the statement, or in subroutines called from it.)
  2. Load config and create a :py:class:`population <population.Population>` - here, the variable ``p``.
  3. If required, create and add :py:mod:`reporters <reporting>`.
  4. Create a :py:class:`DistributedEvaluator(addr_of_primary_node, b'some_password', eval_function, mode=MODE_AUTO) <distributed.DistributedEvaluator>` - here, the variable ``de``.
  5. Call :py:meth:`de.start(exit_on_stop=True) <distributed.DistributedEvaluator.start>`. The ``start()`` call will block on the secondary nodes and call :pylib:`sys.exit(0) <sys.html#sys.exit>` when the NEAT evolution finishes. This means that the following code will only be executed on the primary node.
  6. Start the evaluation using :py:meth:`p.run(de.evaluate, number_of_generations) <population.Population.run>`.
  7. Stop the secondary nodes using :py:meth:`de.stop() <distributed.DistributedEvaluator.stop>`.
  8. You are done. You may want to save the winning genome(s) or show some :py:mod:`statistics`.

  See :file:`examples/xor/evolve-feedforward-distributed.py` for a complete example.

  .. note::

    The below contains some (but not complete) information about private functions, classes, and similar (starting with ``_``); this documentation is meant to help with
    maintaining and improving the code, not for enabling external use, and the interface may change **rapidly** with no warning.

  .. py:data:: MODE_AUTO
  .. py:data:: MODE_PRIMARY
  .. py:data:: MODE_SECONDARY

    Values - which should be treated as constants - that are used for the ``mode`` argument of :py:class:`DistributedEvaluator`. If MODE_AUTO,
    :py:func:`_determine_mode()` uses :py:func:`host_is_local()` and the specified ``addr`` of the :term:`primary node` to decide the mode; the other two specify it.

  .. py:data:: _STATE_RUNNING
  .. py:data:: _STATE_SHUTDOWN
  .. py:data:: _STATE_FORCED_SHUTDOWN

    Values - which should be treated as constants - that are used to determine the current state (whether the secondaries should be continuing the run or not).

  .. py:exception:: ModeError(RuntimeError)

    An exception raised when a mode-specific method is being called without being in the mode - either a primary-specific method
    called by a :term:`secondary node` or a secondary-specific method called by a :term:`primary node`.

  .. py:function:: host_is_local(hostname, port=22)

    Returns True if the hostname points to the localhost (including shares addresses), otherwise False.

    :param str hostname: The hostname to be checked; will be put through `socket.getfqdn`.
    :param port: The optional port for `socket` functions requiring one. Defaults to 22, the ssh port.
    :type port: :pytypes:`int <typesnumeric>`
    :return: Whether the hostname appears to be equivalent to that of the localhost.
    :rtype: :pytypes:`bool <typesnumeric>`

  .. py:function:: _determine_mode(addr, mode)

    Returns the mode that should be used.  If ``mode`` is :py:data:`MODE_AUTO`, this is determined by checking (via :py:func:`host_is_local()`) if ``addr`` points
    to the localhost; if it does, it returns :py:data:`MODE_PRIMARY`, else it returns :py:data:`MODE_SECONDARY`. If mode is either MODE_PRIMARY or
    MODE_SECONDARY, it returns the ``mode`` argument. Otherwise, a ValueError is raised.

    :param addr: Either a tuple of (hostname, port) pointing to the machine that has the :term:`primary node`, or the hostname (as ``bytes`` if on 3.X).
    :type addr: tuple(str, int) or bytes
    :param int mode: Specifies the mode to run in - must be one of :py:data:`MODE_AUTO`, :py:data:`MODE_PRIMARY`, or :py:data:`MODE_SECONDARY`.
    :raises ValueError: If the mode is not one of the above.

  .. py:function:: chunked(data, chunksize)

     Splits up ``data`` and returns it as a list of chunks containing at most ``chunksize`` elements of data.

    :param data: The data to split up; takes any :pygloss:`iterable`.
    :type data: list(object) or tuple(object) or set(object)
    :param chunksize: The maximum number of elements per chunk.
    :type chunksize: :pytypes:`int <typesnumeric>`
    :return: A list of chunks containing (as a list) at most ``chunksize`` elements of data.
    :rtype: list(list(object))
    :raises ValueError: If ``chunksize`` is not 1+ or is not an integer

  .. py:class:: _ExtendedManager(addr, authkey, mode, start=False)

    Manages the :pylib:`multiprocessing.managers.SyncManager <multiprocessing.html#multiprocessing.managers.SyncManager>` instance. Initializes
    ``self._secondary_state`` to :py:data:`_STATE_RUNNING`.

    :param addr: Should be a tuple of (hostname, port) pointing to the machine running the DistributedEvaluator in primary mode. If mode is :py:data:`MODE_AUTO`, the mode is determined by checking whether the hostname points to this host or not (via :py:func:`_determine_mode()` and :py:func:`host_is_local()`).
    :type addr: tuple(str, int)
    :param authkey:  The password used to restrict access to the manager. All DistributedEvaluators need to use the same authkey. Note that this needs to be a :pytypes:`bytes` object for Python 3.X, and should be in 2.7 for compatibility (identical in 2.7 to a `str` object). For more information, see under :py:class:`DistributedEvaluator`.
    :type authkey: :pytypes:`bytes`
    :param int mode: Specifies the mode to run in - must be one of :py:data:`MODE_AUTO`, :py:data:`MODE_PRIMARY`, or :py:data:`MODE_SECONDARY`. Processed by :py:func:`_determine_mode()`.
    :param bool start: Whether to call the :py:meth:`start()` method after initialization.

    .. py:method:: __reduce__()

      Used by `pickle` to serialize instances of this class. TODO: Appears to assume that ``start`` (for initialization) should be true; perhaps ``self.manager``
      should be checked? (This may require :py:meth::`stop()` to set ``self.manager`` to ``None``, incidentally.)

      :return: Information about the class instance; a tuple of (class name, tuple(addr, authkey, mode, True)).
      :rtype: tuple(str, tuple(tuple(str, int), bytes, int, bool))

    .. py:method:: start()

      Starts (if in :py:data:`MODE_PRIMARY`) or connects to (if in :py:data:`MODE_SECONDARY`) the manager.

    .. py:method:: stop()

      Stops the manager using :pylib:`shutdown <multiprocessing.html#multiprocessing.managers.BaseManager.shutdown>` .
      TODO: Should this set ``self.manager`` to None?

    .. py:method:: set_secondary_state(value)

      Sets the value for the ``secondary_state``, shared between the nodes via :pylib:`multiprocessing.managers.Value <multiprocessing.html#multiprocessing.managers.SyncManager.Value>`.

      :param int value: The desired secondary state; must be one of :py:data:`_STATE_RUNNING`, :py:data:`_STATE_SHUTDOWN`, or :py:data:`_STATE_FORCED_SHUTDOWN`.
      :raises ValueError: If the ``value`` is not one of the above.
      :raises RuntimeError: If the manager has not been :py:meth:`started <start()>`.

    .. py:attribute:: secondary_state

      The :pylib:`property <functions.html#property>` ``secondary_state`` - whether the secondary nodes should still be processing elements.

    .. py:method:: get_inqueue()

      Returns the inqueue.

      :return: The incoming :pylib:`queue <multiprocessing.html#multiprocessing.Queue>`.
      :rtype: :datamodel:`instance <index-48>`
      :raises RuntimeError: If the manager has not been :py:meth:`started <start()>`.

    .. py:method:: get_outqueue()

      Returns the outqueue.

      :return: The outgoing :pylib:`queue <multiprocessing.html#multiprocessing.Queue>`.
      :rtype: :datamodel:`instance <index-48>`
      :raises RuntimeError: If the manager has not been :py:meth:`started <start()>`.

    .. py:method:: get_namespace()

      Returns the manager's namespace instance.

      :return: The :pylib:`namespace <argparse.html#argparse.Namespace>`.
      :rtype: :datamodel:`instance <index-48>`
      :raises RuntimeError: If the manager has not been :py:meth:`started <start()>`.


  .. index:: fitness function
  .. index:: fitness

  .. py:class:: DistributedEvaluator(addr, authkey, eval_function, secondary_chunksize=1, num_workers=None, worker_timeout=60, mode=MODE_AUTO)

    An evaluator working across multiple machines (:term:`compute nodes <compute node>`).

    .. warning::

      See :pylib:`Authentication Keys <multiprocessing.html#authentication-keys>` for more on the ``authkey`` parameter, used to restrict access to the manager.

    :param addr: Should be a tuple of (hostname, port) pointing to the machine running the DistributedEvaluator in primary mode. If mode is :py:data:`MODE_AUTO`, the mode is determined by checking whether the hostname points to this host or not (via :py:func:`host_is_local()`).
    :type addr: tuple(str, int)
    :param authkey:  The password used to restrict access to the manager. All DistributedEvaluators need to use the same authkey. Note that this needs to be a :pytypes:`bytes` object for Python 3.X, and should be in 2.7 for compatibility (identical in 2.7 to a `str` object).
    :type authkey: :pytypes:`bytes`
    :param eval_function: The eval_function should take two arguments - a genome object and a config object - and return a single :pytypes:`float <typesnumeric>` (the genome's fitness) Note that this is not the same as how a fitness function is called by :py:meth:`Population.run <population.Population.run>`, nor by :py:class:`ParallelEvaluator <parallel.ParallelEvaluator>` (although it is more similar to the latter).
    :type eval_function: `function`
    :param secondary_chunksize: The number of :term:`genomes <genome>` that will be sent to a :term:`secondary node` at any one time.
    :type secondary_chunksize: :pytypes:`int <typesnumeric>`
    :param num_workers: The number of worker processes per :term:`secondary node`, used for evaluating genomes. If None, will use :pylib:`multiprocessing.cpu_count() <multiprocessing.html#multiprocessing.cpu_count>`  to determine the number of processes (see further below regarding this default). If 1 (for a secondary node), including if there is no usable result from ``multiprocessing.cpu_count()``, then the process creating the DistributedEvaluator instance will also do the evaluations.
    :type num_workers: :pytypes:`int <typesnumeric>` or None
    :param worker_timeout:  specifies the timeout (in seconds) for a secondary node getting the results from a worker subprocess; if None, there is no timeout.
    :type worker_timeout: :pytypes:`float <typesnumeric>` or None
    :param int mode: Specifies the mode to run in - must be one of :py:data:`MODE_AUTO` (the default), :py:data:`MODE_PRIMARY`, or :py:data:`MODE_SECONDARY`.
    :raises ValueError: If the mode is not one of the above.

    .. note::

      Whether the default for ``num_workers`` is appropriate can vary depending on the evaluation function (e.g., whether cpu-bound, memory-bound, i/o-bound...), python implementation, and other factors; if unsure and maximal per-machine performance is critical, experimentation will be required.

    .. py:method:: is_primary()

      Returns True if the caller is the :term:`primary node`; otherwise False.

      :return: `True` if primary, `False` if :term:`secondary <secondary node>`
      :rtype: :pytypes:`bool <typesnumeric>`

    .. py:method:: is_master()

      A backward-compatibility wrapper for :py:meth:`is_primary`.

      :return: `True` if primary, `False` if :term:`secondary <secondary node>`
      :rtype: :pytypes:`bool <typesnumeric>`
      :raises DeprecationWarning: Always.

      .. deprecated:: 0.92

    .. py:method:: start(exit_on_stop=True, secondary_wait=0, reconnect=False)

      If the DistributedEvaluator is in primary mode, starts the manager process and returns. If the DistributedEvaluator is in secondary mode, it connects to the
      manager and waits for tasks.

      :param exit_on_stop: If a secondary node, whether to exit if (unless ``reconnect`` is ``True``) the connection is lost, the primary calls for a shutdown (via :py:meth:`stop()`), or - even if ``reconnect`` is True - the primary calls for a forced shutdown (via calling :py:meth:`stop()` with ``force_secondary_shutdown`` set to ``True``).
      :type exit_on_stop: :pytypes:`bool <typesnumeric>`
      :param secondary_wait: Specifies the time (in seconds) to sleep before actually starting, if a :term:`secondary node`.
      :type secondary_wait: :pytypes:`float <typesnumeric>`
      :param bool reconnect: If a secondary node, whether it should try to reconnect if the connection is lost.
      :raises RuntimeError: If already started.
      :raises ValueError: If the mode is invalid.

    .. py:method:: stop(wait=1, shutdown=True, force_secondary_shutdown=False)

      Stops all secondaries.

      :param wait: Time (in seconds) to wait after telling the secondaries to stop.
      :type wait: :pytypes:`float <typesnumeric>`
      :param shutdown: Whether to :pylib:`shutdown <multiprocessing.html#multiprocessing.managers.BaseManager.shutdown>` the :pylib:`multiprocessing.managers.SyncManager <multiprocessing.html#multiprocessing.managers.SyncManager>` also (after the wait, if any).
      :type shutdown: :pytypes:`bool <typesnumeric>`
      :param bool force_secondary_shutdown: Causes secondaries to shutdown even if started with ``reconnect`` true (via setting the ``secondary_state`` to :py:data:`_STATE_FORCED_SHUTDOWN` instead of :py:data:`_STATE_SHUTDOWN`).
      :raises ModeError: If not the :term:`primary node` (not in :py:data:`MODE_PRIMARY`).
      :raises RuntimeError: If not yet :py:meth:`started <start()>`.

    .. py:method:: evaluate(genomes, config)

      Evaluates the genomes. Distributes the genomes to the secondary nodes, then gathers the fitnesses from the secondary nodes and assigns them to the
      genomes. Must not be called by :term:`secondary nodes <secondary node>`. TODO: Improved handling of errors from broken connections with
      the secondary nodes may be needed.

      :param genomes: Dictionary of (:term:`genome_id <key>`, genome) 
      :type genomes: dict(int, :datamodel:`instance <index-48>`)
      :param config: Configuration object.
      :type config: :datamodel:`instance <index-48>`
      :raises ModeError: If not the :term:`primary node` (not in :py:data:`MODE_PRIMARY`).

  .. versionadded:: 0.92

.. py:module:: genes
   :synopsis: Handles node and connection genes.

genes
--------
Handles node and connection genes.

  .. inheritance-diagram:: genes iznn.IZNodeGene

  .. index:: key
  .. index:: ! gene

  .. py:class:: BaseGene(key)

    Handles functions shared by multiple types of genes (both :term:`node` and :term:`connection`), including :term:`crossover` and
    calling :term:`mutation` methods.

    :param key: The gene :term:`identifier <key>`. Note: For connection genes, determining whether they are :term:`homologous` (for :term:`genomic distance` and :term:`crossover` determination) uses the (ordered) identifiers of the connected nodes.
    :type key: :pytypes:`int <typesnumeric>` or tuple(int, int)

    .. py:method:: __str__()

      Converts gene attributes into a printable format.

      :return: Stringified gene instance.
      :rtype: str

    .. py:method:: __lt__(other)

      Allows sorting genes by :term:`keys <key>`.

      :param other: The other `BaseGene` instance.
      :type other: :datamodel:`instance <index-48>`
      :return: Whether the calling instance's key is less than that of the ``other`` instance.
      :rtype: :pytypes:`bool <typesnumeric>`

    .. py:classmethod:: parse_config(config, param_dict)

      Placeholder; parameters are entirely in gene :term:`attributes`.

    .. py:classmethod:: get_config_params()

      Fetches configuration parameters from each gene class' ``_gene_attributes`` list (using
      :py:meth:`BaseAttribute.get_config_params <attributes.BaseAttribute.get_config_params>`).
      Used by :py:class:`genome.DefaultGenomeConfig` to include gene parameters in its configuration parameters.

      :return: List of configuration parameters (as :py:class:`config.ConfigParameter` instances) for the gene attributes.
      :rtype: list(:datamodel:`instance <index-48>`)
      :raises DeprecationWarning: If the gene class uses ``__gene_attributes__`` instead of ``_gene_attributes``

    .. py:method:: init_attributes(config)

      Initializes its gene attributes using the supplied configuration object and :py:meth:`FloatAttribute.init_value <attributes.FloatAttribute.init_value>`,
      :py:meth:`BoolAttribute.init_value <attributes.BoolAttribute.init_value>`, or
      :py:meth:`StringAttribute.init_value <attributes.StringAttribute.init_value>` as appropriate.

      :param config: Configuration object to be used by the appropriate :py:mod:`attributes` class.
      :type config: :datamodel:`instance <index-48>`

    .. index::
      see: mutate; mutation
    .. index:: ! mutation

    .. py:method:: mutate(config)

      :term:`Mutates <mutation>` (possibly) its gene attributes using the supplied configuration object and
      :py:meth:`FloatAttribute.init_value <attributes.FloatAttribute.mutate_value>`,
      :py:meth:`BoolAttribute.init_value <attributes.BoolAttribute.mutate_value>`, or
      :py:meth:`StringAttribute.init_value <attributes.StringAttribute.mutate_value>` as appropriate.

      :param config: Configuration object to be used by the appropriate :py:mod:`attributes` class.
      :type config: :datamodel:`instance <index-48>`

    .. py:method:: copy()

      Makes a copy of itself, including its subclass, :term:`key`, and all gene attributes.

      :return: A copied gene
      :rtype: :datamodel:`instance <index-48>`

    .. index:: ! crossover

    .. py:method:: crossover(gene2)

      Creates a new gene via :term:`crossover` - randomly inheriting attributes from its parents. The two genes must be :term:`homologous`, having
      the same :term:`key`/id.

      :param gene2: The other gene.
      :type gene2: :datamodel:`instance <index-48>`
      :return: A new gene, with the same key/id, with other attributes being copied randomly (50/50 chance) from each parent gene.
      :rtype: :datamodel:`instance <index-48>`

  .. index:: node
  .. index:: ! genetic distance
  .. index:: genomic distance
  .. index:: ! compatibility_weight_coefficient

  .. py:class:: DefaultNodeGene(BaseGene)

    Groups :py:mod:`attributes` specific to :term:`node` genes - such as :term:`bias` - and calculates
    genetic distances between two :term:`homologous` (not :term:`disjoint` or excess) node genes.

    .. py:method:: distance(other, config)

      Determines the degree of differences between node genes using their 4 :term:`attributes`;
      the final result is multiplied by the configured :ref:`compatibility_weight_coefficient <compatibility-weight-coefficient-label>`.

      :param other: The other ``DefaultNodeGene``.
      :type other: :datamodel:`instance <index-48>`
      :param config: The genome configuration object.
      :type config: :datamodel:`instance <index-48>`
      :return: The contribution of this pair to the :term:`genomic distance` between the source genomes.
      :rtype: :pytypes:`float <typesnumeric>`

  .. index:: connection
  .. index:: ! genetic distance
  .. index:: genomic distance
  .. index:: ! compatibility_weight_coefficient

  .. py:class:: DefaultConnectionGene(BaseGene)

    Groups :py:mod:`attributes` specific to :term:`connection` genes - such as :term:`weight` - and calculates
    genetic distances between two :term:`homologous` (not :term:`disjoint` or excess) connection genes.

    .. py:method:: distance(other, config)

      Determines the degree of differences between connection genes using their 2 :term:`attributes`;
      the final result is multiplied by the configured :ref:`compatibility_weight_coefficient <compatibility-weight-coefficient-label>`.

      :param other: The other ``DefaultConnectionGene``.
      :type other: :datamodel:`instance <index-48>`
      :param config: The genome configuration object.
      :type config: :datamodel:`instance <index-48>`
      :return: The contribution of this pair to the :term:`genomic distance` between the source genomes.
      :rtype: :pytypes:`float <typesnumeric>`

    .. versionchanged:: 0.92
      ``__gene_attributes__`` changed to ``_gene_attributes``, since it is not a Python internal variable. Updates also made due to addition of
      default capabilities to :py:mod:`attributes`.

.. py:module:: genome
   :synopsis: Handles genomes (individuals in the population).

genome
-----------
Handles genomes (individuals in the population).

  .. inheritance-diagram:: genome iznn.IZGenome

  .. index:: initial_connection
  .. index:: compatibility_disjoint_coefficient
  .. index:: compatibility_weight_coefficient
  .. index:: conn_add_prob
  .. index:: conn_delete_prob
  .. index:: node_add_prob
  .. index:: node_delete_prob
  .. index:: structural_mutation_surer
  .. index:: single_structural_mutation
  .. index:: feed_forward
  .. index:: num_hidden
  .. index:: num_outputs
  .. index:: num_inputs

  .. py:class:: DefaultGenomeConfig(params)

    Does the configuration for the DefaultGenome class. Has the `list <list>` ``allowed_connectivity``, which defines the available
    values for :ref:`initial_connection <initial-connection-config-label>`. Includes parameters taken from the configured gene classes, such
    as :py:class:`genes.DefaultNodeGene`, :py:class:`genes.DefaultConnectionGene`, or :py:class:`iznn.IZNodeGene`. The
    :py:class:`activations.ActivationFunctionSet` instance is available via its ``activation_defs`` attribute, and the
    :py:class:`aggregations.AggregationFunctionSet` instance is available via its ``aggregation_defs`` - or, for compatibility,
    ``aggregation_function_defs`` - attributes. TODO: Check for unused configuration parameters from the config file.

    :param params: Parameters from configuration file and DefaultGenome initialization (by parse_config).
    :type params: dict(str, str)
    :raises RuntimeError: If ``initial_connection`` or :ref:`structural_mutation_surer <structural-mutation-surer-label>` is invalid.

    .. versionchanged:: 0.92
      Aggregation functions moved to :py:mod:`aggregations`; additional configuration parameters added.

    .. index:: ! activation function

    .. py:method:: add_activation(name, func)

      Adds a new :term:`activation function`, as described in :ref:`customization-label`.
      Uses :py:meth:`ActivationFunctionSet.add <activations.ActivationFunctionSet.add>`.

      :param str name: The name by which the function is to be known in the :ref:`configuration file <activation-function-config-label>`.
      :param func: A function meeting the requirements of :py:func:`activations.validate_activation`.
      :type func: `function`

    .. index:: ! aggregation function

    .. py:method:: add_aggregation(name, func)

      Adds a new :term:`aggregation function`.
      Uses :py:meth:`AggregationFunctionSet.add <aggregations.AggregationFunctionSet.add>`.

      :param str name: The name by which the function is to be known in the :ref:`configuration file <aggregation-function-config-label>`.
      :param func: A function meeting the requirements of :py:func:`aggregations.validate_aggregation`.
      :type func: `function`

      .. versionadded:: 0.92

    .. py:method:: save(f)

      Saves the :ref:`initial_connection <initial-connection-config-label>` configuration and uses :py:func:`config.write_pretty_params` to write out the
      other parameters.

      :param f: The file object to be written to.
      :type f: :pygloss:`file <file-object>`
      :raises RuntimeError: If the value for a :ref:`partial-connectivity configuration <initial-connection-config-label>` is not in [0.0,1.0].

    .. index:: ! key

    .. py:method:: get_new_node_key(node_dict)

      Finds the next unused node :term:`key`. TODO: Explore using the same :term:`node` key if a particular connection is replaced in more than
      one genome in the same generation (use a :py:meth:`reporting.BaseReporter.end_generation` method to wipe a dictionary of connection tuples
      versus node keys).

      :param node_dict: A dictionary of node keys vs nodes
      :type node_dict: dict(int, :datamodel:`instance <index-48>`)
      :return: A currently-unused node key.
      :rtype: :pytypes:`int <typesnumeric>`
      :raises AssertionError: If a newly-created id is already in the node_dict.

      .. versionchanged:: 0.92
        Moved from DefaultGenome so no longer only single-genome-instance unique.

    .. index:: structural_mutation_surer
    .. index:: single_structural_mutation

    .. py:method:: check_structural_mutation_surer()

      Checks vs :ref:`structural_mutation_surer <structural-mutation-surer-label>` and, if necessary, ``single_structural_mutation`` to decide if
      changes from the former should happen.

      :returns: If should have a structural mutation under a wider set of circumstances.
      :rtype: :pytypes:`bool <typesnumeric>`

      .. versionadded:: 0.92

  .. index:: key
  .. index:: ! pin

  .. py:class:: DefaultGenome(key)

    A :term:`genome` for generalized neural networks. For class requirements, see :ref:`genome-interface-label`.
    Terminology:
    :term:`pin` - Point at which the network is conceptually connected to the external world; pins are either input or output.
    :term:`node` - Analog of a physical neuron.
    :term:`connection` - Connection between a pin/node output and a node's input, or between a node's output and a pin/node input.
    :term:`key` - Identifier for an object, unique within the set of similar objects.
    Design assumptions and conventions.
    1. Each output pin is connected only to the output of its own unique :term:`neuron <output node>` by an implicit connection with weight one. This connection is permanently enabled.
    2. The output pin's key is always the same as the key for its associated neuron.
    3. Output neurons can be modified but not deleted.
    4. The input values are applied to the :term:`input pins <input node>` unmodified.

    :param int key: :term:`Identifier <key>` for this individual/genome.

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Provides default :term:`node` and :term:`connection` :term:`gene` specifications (from :py:mod:`genes`) and
      uses `DefaultGenomeConfig` to do the rest of the configuration.

      :param param_dict: Dictionary of parameters from configuration file.
      :type param_dict: dict(str, str)
      :return: Configuration object; considered opaque by rest of code, so type may vary by implementation (here, a `DefaultGenomeConfig` instance).
      :rtype: :datamodel:`instance <index-48>`

    .. py:classmethod:: write_config(f, config)

      Required interface method. Saves configuration using :py:meth:`DefaultGenomeConfig.save`.

      :param f: File object to write to.
      :type f: :pygloss:`file <file-object>`
      :param config: Configuration object (here, a `DefaultGenomeConfig` instance).
      :type config: :datamodel:`instance <index-48>`

    .. index:: ! initial_connection
    .. index:: hidden node
    .. index:: input node
    .. index:: output node

    .. py:method:: configure_new(config)

      Required interface method. Configures a new genome (itself) based on the given
      configuration object, including genes for :term:`connectivity <connection>` (based on :ref:`initial_connection <initial-connection-config-label>`) and
      starting :term:`nodes <node>` (as defined by :term:`num_hidden <hidden node>`, :term:`num_inputs <input node>`, and
      :term:`num_outputs <output node>` in the :ref:`configuration file <num-nodes-config-label>`.

      :param config: Genome configuration object.
      :type config: :datamodel:`instance <index-48>`

    .. index:: ! crossover

    .. py:method:: configure_crossover(genome1, genome2, config)

      Required interface method. Configures a new genome (itself) by :term:`crossover` from two parent genomes. :term:`disjoint`
      or :term:`excess` genes are inherited from the fitter of the two parents, while :term:`homologous` genes use the gene class' crossover function
      (e.g., :py:meth:`genes.BaseGene.crossover`).

      :param genome1: The first parent genome.
      :type genome1: :datamodel:`instance <index-48>`
      :param genome2: The second parent genome.
      :type genome2: :datamodel:`instance <index-48>`
      :param config: Genome configuration object; currently ignored.
      :type config: :datamodel:`instance <index-48>`

    .. index:: ! mutation
    .. index:: ! single_structural_mutation
    .. index:: node_add_prob
    .. index:: node_delete_prob
    .. index:: conn_add_prob
    .. index:: conn_delete_prob

    .. py:method:: mutate(config)

      Required interface method. :term:`Mutates <mutation>` this genome. What mutations take place are determined by configuration file settings, such
      as :ref:`node_add_prob <node-add-prob-label>` and ``node_delete_prob`` for the likelihood of adding or removing a :term:`node` and
      :ref:`conn_add_prob <conn-add-prob-label>` and ``conn_delete_prob`` for the likelihood of adding or removing a :term:`connection`. Checks
      :ref:`single_structural_mutation <structural-mutation-surer-label>` for whether more than one structural mutation should be permitted per call.
      Non-structural mutations (to gene :term:`attributes`) are performed by calling the appropriate ``mutate`` method(s) for
      connection and node genes (generally :py:meth:`genes.BaseGene.mutate`).

      :param config: Genome configuration object.
      :type config: :datamodel:`instance <index-48>`

      .. versionchanged:: 0.92
        ``single_structural_mutation`` config parameter added.

    .. index:: node
    .. index:: structural_mutation_surer
    .. index:: check_structural_mutation_surer()

    .. py:method:: mutate_add_node(config)

      Takes a randomly-selected existing connection, turns its :term:`enabled` attribute to ``False``, and makes two new (enabled) connections with a
      new :term:`node` between them, which join the now-disabled connection's nodes. The connection weights are chosen so as to potentially have
      roughly the same behavior as the original connection, although this will depend on the :term:`activation function`, :term:`bias`, and
      :term:`response` multiplier of the new node. If there are no connections available, may call :py:meth:`mutate_add_connection` instead,
      depending on the result from :py:meth:`check_structural_mutation_surer <genome.DefaultGenomeConfig.check_structural_mutation_surer>`.

      :param config: Genome configuration object.
      :type config: :datamodel:`instance <index-48>`

      .. versionchanged:: 0.92
        Potential addition of connection instead added.

    .. index:: ! connection

    .. py:method:: add_connection(config, input_key, output_key, weight, enabled)

      Adds a specified new connection; its :term:`key` is the `tuple` of ``(input_key, output_key)``. TODO: Add further validation of this connection addition?

      :param config: Genome configuration object.
      :type config: :datamodel:`instance <index-48>`
      :param int input_key: :term:`Key <key>` of the connection's input-side node.
      :param int output_key: Key of the connection's output-side node.
      :param float weight: The :term:`weight` the new connection should have.
      :param bool enabled: The :term:`enabled` attribute the new connection should have.

    .. index:: ! feed_forward
    .. index:: connection
    .. index:: structural_mutation_surer
    .. index:: check_structural_mutation_surer()

    .. py:method:: mutate_add_connection(config)

      Attempts to add a randomly-selected new connection, with some filtering:
      1. :term:`input nodes <input node>` cannot be at the output end.
      2. Existing connections cannot be duplicated. (If an existing connection is selected, it may be :term:`enabled` depending on the result from :py:meth:`check_structural_mutation_surer <genome.DefaultGenomeConfig.check_structural_mutation_surer>`.)
      3. Two :term:`output nodes <output node>` cannot be connected together.
      4. If :ref:`feed_forward <feed-forward-config-label>` is set to ``True`` in the configuration file, connections cannot create :py:func:`cycles <graphs.creates_cycle>`.

      :param config: Genome configuration object
      :type config: :datamodel:`instance <index-48>`

      .. versionchanged:: 0.92
        Output nodes not allowed to be connected together. Possibility of enabling existing connection added.

    .. py:method:: mutate_delete_node(config)

      Deletes a randomly-chosen (non-:term:`output <output node>`/input) node along with its connections.

      :param config: Genome configuration object
      :type config: :datamodel:`instance <index-48>`

    .. py:method:: mutate_delete_connection()

      Deletes a randomly-chosen connection. TODO: If the connection is :term:`enabled`, have an option to - possibly with a :term:`weight`-dependent
      chance - turn its :term:`enabled` attribute to ``False`` instead.

    .. index:: ! compatibility_disjoint_coefficient
    .. index:: ! genomic distance
    .. index:: genetic distance

    .. py:method:: distance(other, config)

      Required interface method. Returns the :term:`genomic distance` between this genome and the other.
      This distance value is used to compute genome compatibility for :py:mod:`speciation <species>`. Uses (by default) the
      :py:meth:`genes.DefaultNodeGene.distance` and :py:meth:`genes.DefaultConnectionGene.distance` methods for
      :term:`homologous` pairs, and the configured :ref:`compatibility_disjoint_coefficient <compatibility-disjoint-coefficient-label>` for
      disjoint/excess genes. (Note that this is one of the most time-consuming portions of the library; optimization - such as using
      `cython <http://cython.org>`_ - may be needed if using an unusually fast fitness function and/or an unusually large population.)

      :param other: The other DefaultGenome instance (genome) to be compared to.
      :type other: :datamodel:`instance <index-48>`
      :param config: The genome configuration object.
      :type config: :datamodel:`instance <index-48>`
      :return: The genomic distance.
      :rtype: :pytypes:`float <typesnumeric>`

    .. py:method:: size()

      Required interface method. Returns genome ``complexity``, taken to be (number of nodes, number of enabled connections); currently only used
      for reporters - some retrieve this information for the highest-fitness genome at the end of each generation.

      :return: Genome complexity
      :rtype: tuple(int, int)

    .. py:method:: __str__()

      Gives a listing of the genome's nodes and connections.

      :return: Node and connection information.
      :rtype: str

    .. index:: node

    .. py:staticmethod:: create_node(config, node_id)

      Creates a new node with the specified :term:`id <key>` (including for its :term:`gene`), using the specified configuration object to retrieve the proper
      node gene type and how to initialize its attributes.

      :param config: The genome configuration object.
      :type config: :datamodel:`instance <index-48>`
      :param int node_id: The key for the new node.
      :return: The new node instance.
      :rtype: :datamodel:`instance <index-48>`

    .. index:: connection

    .. py:staticmethod:: create_connection(config, input_id, output_id)

      Creates a new connection with the specified :term:`id <key>` pair as its key (including for its :term:`gene`, as a `tuple`), using the specified
      configuration object to retrieve the proper connection gene type and how to initialize its attributes.

      :param config: The genome configuration object.
      :type config: :datamodel:`instance <index-48>`
      :param int input_id: The input end node's key.
      :param int output_id: The output end node's key.
      :return: The new connection instance.
      :rtype: :datamodel:`instance <index-48>`

    .. index:: ! initial_connection

    .. py:method:: connect_fs_neat_nohidden(config)

      Connect one randomly-chosen input to all :term:`output nodes <output node>` (FS-NEAT without connections to :term:`hidden nodes <hidden node>`,
      if any). Previously called ``connect_fs_neat``. Implements the ``fs_neat_nohidden`` setting for :ref:`initial_connection <initial-connection-config-label>`.

      :param config: The genome configuration object.
      :type config: :datamodel:`instance <index-48>`

      .. versionchanged:: 0.92
        Connect_fs_neat, connect_full, connect_partial split up - documentation vs program conflict.

    .. py:method:: connect_fs_neat_hidden(config)

      Connect one randomly-chosen input to all :term:`hidden nodes <hidden node>` and :term:`output nodes <output node>` (FS-NEAT with
      connections to hidden nodes, if any). Implements the ``fs_neat_hidden`` setting for :ref:`initial_connection <initial-connection-config-label>`.

      :param config: The genome configuration object.
      :type config: :datamodel:`instance <index-48>`

      .. versionchanged:: 0.92
        Connect_fs_neat, connect_full, connect_partial split up - documentation vs program conflict.

    .. py:method:: compute_full_connections(config, direct)

      Compute connections for a fully-connected feed-forward genome--each input connected to all hidden nodes (and output nodes if ``direct`` is set or
      there are no hidden nodes), each hidden node connected to all output nodes. (Recurrent genomes will also include node self-connections.)

      :param config: The genome configuration object.
      :type config: :datamodel:`instance <index-48>`
      :param bool direct: Whether or not, if there are :term:`hidden nodes <hidden node>`, to include links directly from input to output.
      :return: The list of connections, as (input :term:`key`, output key) tuples
      :rtype: list(tuple(int,int))

      .. versionchanged:: 0.92
        "Direct" added to help with documentation vs program conflict for ``initial_connection`` of ``full`` or ``partial``.

    .. py:method:: connect_full_nodirect(config)

      Create a fully-connected genome (except no direct :term:`input <input node>` to :term:`output <output node>` connections unless there are no
      :term:`hidden nodes <hidden node>`).

      :param config: The genome configuration object.
      :type config: :datamodel:`instance <index-48>`

      .. versionchanged:: 0.92
        Connect_fs_neat, connect_full, connect_partial split up - documentation vs program conflict.

    .. py:method:: connect_full_direct(config)

      Create a fully-connected genome, including direct input-output connections even if there are hidden nodes.

      :param config: The genome configuration object.
      :type config: :datamodel:`instance <index-48>`

      .. versionchanged:: 0.92
        Connect_fs_neat, connect_full, connect_partial split up - documentation vs program conflict.

    .. py:method:: connect_partial_nodirect(config)

      Create a partially-connected genome, with (unless there are no :term:`hidden nodes <hidden node>`) no direct input-output connections.

      :param config: The genome configuration object.
      :type config: :datamodel:`instance <index-48>`

      .. versionchanged:: 0.92
        Connect_fs_neat, connect_full, connect_partial split up - documentation vs program conflict.

    .. py:method:: connect_partial_direct(config)

      Create a partially-connected genome, possibly including direct input-output connections even if there are hidden nodes.

      :param config: The genome configuration object.
      :type config: :datamodel:`instance <index-48>`

      .. versionchanged:: 0.92
        Connect_fs_neat, connect_full, connect_partial split up - documentation vs program conflict.

.. index:: feed_forward
.. index:: feedforward
.. index::
  see: feed-forward; feedforward
.. index:: recurrent

.. py:module:: graphs
   :synopsis: Directed graph algorithm implementations.

graphs
---------
Directed graph algorithm implementations.

  .. py:function:: creates_cycle(connections, test)

    Returns true if the addition of the ``test`` :term:`connection` would create a cycle, assuming that no cycle already exists in the graph represented
    by ``connections``. Used to avoid :term:`recurrent` networks when a purely :term:`feed-forward` network is desired (e.g., as determined by the
    ``feed_forward`` setting in the :ref:`configuration file <feed-forward-config-label>`.

    :param connections: The current network, as a list of (input, output) connection :term:`identifiers <key>`.
    :type connections: list(tuple(int, int))
    :param test: Possible connection to be checked for causing a cycle.
    :type test: tuple(int, int)
    :return: True if a cycle would be created; false if not.
    :rtype: :pytypes:`bool <typesnumeric>`

  .. py:function:: required_for_output(inputs, outputs, connections)

    Collect the :term:`nodes <node>` whose state is required to compute the final network output(s).

    :param inputs: the :term:`input node` :term:`identifiers <key>`; **it is assumed that the input identifier set and the node identifier set are disjoint.**
    :type inputs: list(int)
    :param outputs: the :term:`output node` identifiers; by convention, the output node :term:`ids <key>` are always the same as the output index.
    :type outputs: list(int)
    :param connections: list of (input, output) connections in the network; should only include enabled ones.
    :type connections: list(tuple(int, int))
    :return: A set of node identifiers.
    :rtype: set(int)

  .. py:function:: feed_forward_layers(inputs, outputs, connections)

    Collect the layers whose members can be evaluated in parallel in a :term:`feed-forward` network.

    :param inputs: the network :term:`input node` :term:`identifiers <key>`.
    :type inputs: list(int)
    :param outputs: the :term:`output node` :term:`identifiers <key>`.
    :type outputs: list(int)
    :param connections: list of (input, output) connections in the network; should only include enabled ones.
    :type connections: list(tuple(int, int))
    :return: A list of layers, with each layer consisting of a set of :term:`identifiers <key>`; only includes nodes returned by `required_for_output`.
    :rtype: list(set(int))

.. py:module:: iznn
   :synopsis: Implements a spiking neural network (closer to in vivo neural networks) based on Izhikevich's 2003 model.

iznn
------
This module implements a spiking neural network. Neurons are based on the model described by::

  Izhikevich, E. M.
  Simple Model of Spiking Neurons
  IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 14, NO. 6, NOVEMBER 2003

See http://www.izhikevich.org/publications/spikes.pdf.

  .. inheritance-diagram:: iznn

  .. py:data:: REGULAR_SPIKING_PARAMS
  .. py:data:: INTRINSICALLY_BURSTING_PARAMS
  .. py:data:: CHATTERING_PARAMS
  .. py:data:: FAST_SPIKING_PARAMS
  .. py:data:: THALAMO_CORTICAL_PARAMS
  .. py:data:: RESONATOR_PARAMS
  .. py:data:: LOW_THRESHOLD_SPIKING_PARAMS

    Parameter sets (for ``a``, ``b``, ``c``, and ``d``, described below) producing known types of spiking behaviors.

  .. index:: node
  .. index:: gene

  .. py:class:: IZNodeGene(BaseGene)

    Contains attributes for the iznn :term:`node` genes and determines :term:`genomic distances <genomic distance>`.
    TODO: Genomic distance currently does not take into account the node's :term:`bias`.

    .. py:method:: distance(other, config)

      Determines the :term:`genomic distance` between this node gene and the other node gene.

      :param other: The other IZNodeGene instance.
      :type other: :datamodel:`instance <index-48>`
      :param config: Configuration object, in this case a :py:class:`genome.DefaultGenomeConfig` instance.
      :type config: :datamodel:`instance <index-48>`

  .. index:: genome

  .. py:class:: IZGenome(DefaultGenome)

    Contains the parse_config class method for iznn genome configuration, which returns a :py:class:`genome.DefaultGenomeConfig` instance.

  .. py:class:: IZNeuron(bias, a, b, c, d, inputs)

    Sets up and simulates the iznn :term:`nodes <node>` (neurons).

    :param float bias: The bias of the neuron.
    :param float a: The time scale of the recovery variable.
    :param float b: The sensitivity of the recovery variable.
    :param float c: The after-spike reset value of the membrane potential.
    :param float d: The after-spike reset of the recovery variable.
    :param inputs: A list of (input key, weight) pairs for incoming connections.
    :type inputs: list(tuple(int, float))
    :raises RuntimeError: If the number of inputs does not match the number of input nodes.

    .. py:method:: advance(dt_msec)

      Advances simulation time for the neuron by the given time step in milliseconds. TODO: Currently has some numerical stability problems.

      :param float dt_msec: Time step in milliseconds.

    .. py:method:: reset()

      Resets all state variables.

  .. py:class:: IZNN(neurons, inputs, outputs)

    Sets up the network itself and simulates it using the connections and neurons.

    :param neurons: The :py:class:`IZNeuron` instances needed.
    :type neurons: list(:datamodel:`instance <index-48>`)
    :param inputs: The :term:`input node` keys.
    :type inputs: list(int)
    :param outputs: The :term:`output node` keys.
    :type outputs: list(int)

    .. py:method:: set_inputs(inputs)

      Assigns input voltages.

      :param inputs: The input voltages for the :term:`input nodes <input node>`.
      :type inputs: list(:pytypes:`float <typesnumeric>`)

    .. py:method:: reset()

      Resets all neurons to their default state.

    .. py:method:: get_time_step_msec()

      Returns a suggested time step; currently hardwired to 0.05. TODO: Investigate this (particularly effects on numerical stability issues).

      :return: Suggested time step in milliseconds.
      :rtype: :pytypes:`float <typesnumeric>`

    .. py:method:: advance(dt_msec)

      Advances simulation time for all neurons in the network by the input number of milliseconds.

      :param float dt_msec: How many milliseconds to advance the network.
      :return: The values for the :term:`output nodes <output node>`.
      :rtype: list(:pytypes:`float <typesnumeric>`)

    .. py:staticmethod:: create(genome, config)

      Receives a genome and returns its phenotype (a neural network).

      :param genome: An IZGenome instance.
      :type genome: :datamodel:`instance <index-48>`
      :param config: Configuration object, in this implementation a :py:class:`config.Config` instance.
      :type config: :datamodel:`instance <index-48>`
      :return: An IZNN instance.
      :rtype: :datamodel:`instance <index-48>`

    .. versionchanged:: 0.92
      ``__gene_attributes__`` changed to ``_gene_attributes``, since it is not a Python internal variable. 

.. py:module:: math_util
   :synopsis: Contains some mathematical functions not found in the Python2 standard library, plus a mechanism for looking up some commonly used functions (such as for the species_fitness_func) by name.

math_util
-------------
Contains some mathematical/statistical functions not found in the Python2 standard library, plus a mechanism for looking up some commonly used
functions (such as for the :ref:`species_fitness_func <species-fitness-func-label>`) by name.

  .. index:: ! species_fitness_func
  .. index:: stagnation

  .. py:data:: stat_functions

    Lookup table for commonly used ``{value} -> value`` functions, namely `max`, `min`, `mean`, `median`, and `median2`.
    The :ref:`species_fitness_func <species-fitness-func-label>` (used for :py:class:`stagnation.DefaultStagnation`) is required to be one of these.

    .. versionchanged:: 0.92
      `median2` added.

  .. py:function:: mean(values)

    Returns the arithmetic mean.

    :param values: Numbers to take the mean of.
    :type values: list(float) or set(float) or tuple(float)
    :return: The arithmetic mean.
    :rtype: :pytypes:`float <typesnumeric>`

  .. py:function:: median(values)

    Returns the median for odd numbers of values; returns the higher of the middle two values for even numbers of values.

    :param values: Numbers to take the median of.
    :type values: list(float) or set(float) or tuple(float)
    :return: The median.
    :rtype: :pytypes:`float <typesnumeric>`

  .. py:function:: median2(values)

    Returns the median for odd numbers of values; returns the mean of the middle two values for even numbers of values.

    :param values: Numbers to take the median of.
    :type values: list(float) or set(float) or tuple(float)
    :return: The median.
    :rtype: :pytypes:`float <typesnumeric>`

    .. versionadded:: 0.92

  .. py:function:: variance(values)

    Returns the (population) variance.

    :param values: Numbers to get the variance of.
    :type values: list(float) or set(float) or tuple(float)
    :return: The variance.
    :rtype: :pytypes:`float <typesnumeric>`

  .. py:function:: stdev(values)

    Returns the (population) standard deviation. *Note spelling.*

    :param values: Numbers to get the standard deviation of.
    :type values: list(float) or set(float) or tuple(float)
    :return: The standard deviation.
    :rtype: :pytypes:`float <typesnumeric>`

  .. py:function:: softmax(values)

    Compute the softmax (a differentiable/smooth approximization of the maximum function) of the given value set.
    (See the `Wikipedia entry <https://en.wikipedia.org/wiki/Softmax_function>`_ for more on softmax. Envisioned as useful for postprocessing of network output.)

    :param values: Numbers to get the softmax of.
    :type values: list(float) or set(float) or tuple(float)
    :return: :math:`\begin{equation}v_i = \exp(v_i) / s \text{, where } s = \sum(\exp(v_0), \exp(v_1), \dotsc)\end{equation}`
    :rtype: list(:pytypes:`float <typesnumeric>`)

    .. versionchanged:: 0.92
      Previously not functional on Python 3.X due to changes to map.

.. py:module:: nn.feed_forward
   :synopsis: A straightforward feed-forward neural network NEAT implementation.

nn.feed_forward
----------------------

  .. py:class:: FeedForwardNetwork(inputs, outputs, node_evals)

    A straightforward (no pun intended) :term:`feed-forward` neural network NEAT implementation.

    :param inputs: The input :term:`keys <key>` (IDs).
    :type inputs: list(int)
    :param outputs: The output keys.
    :type outputs: list(int)
    :param node_evals: A list of :term:`node` descriptions, with each node represented by a list.
    :type node_evals: list(list(object))

    .. py:method:: activate(inputs)

      Feeds the inputs into the network and returns the resulting outputs.

      :param inputs: The values for the :term:`input nodes <input node>`.
      :type inputs: list(float)
      :return: The values for the :term:`output nodes <output node>`.
      :rtype: list(float)
      :raises RuntimeError: If the number of inputs is not the same as the number of input nodes.

    .. py:staticmethod:: create(genome, config)

      Receives a genome and returns its phenotype.

      :param genome: Genome to return phenotype for.
      :type genome: :datamodel:`instance <index-48>`
      :param config: Configuration object.
      :type config: :datamodel:`instance <index-48>`
      :return: A :py:class:`FeedForwardNetwork` instance.
      :rtype: :datamodel:`instance <index-48>`

.. py:module:: nn.recurrent
   :synopsis: A recurrent (but otherwise straightforward) neural network NEAT implementation.

nn.recurrent
----------------------

  .. py:class:: RecurrentNetwork(inputs, outputs, node_evals)

    A :term:`recurrent` (but otherwise straightforward) neural network NEAT implementation.

    :param inputs: The input :term:`keys <key>` (IDs).
    :type inputs: list(int)
    :param outputs: The output keys.
    :type outputs: list(int)
    :param node_evals: A list of node descriptions, with each node represented by a list.
    :type node_evals: list(list(object))

    .. py:method:: reset()

      Resets all node activations to 0 (necessary due to otherwise retaining state via recurrent connections).

    .. py:method:: activate(inputs)

      Feeds the inputs into the network and returns the resulting outputs.

      :param inputs: The values for the :term:`input nodes <input node>`.
      :type inputs: list(float)
      :return: The values for the :term:`output nodes <output node>`.
      :rtype: list(float)
      :raises RuntimeError: If the number of inputs is not the same as the number of input nodes.

    .. py:staticmethod:: create(genome, config)

      Receives a genome and returns its phenotype.

      :param genome: Genome to return phenotype for.
      :type genome: :datamodel:`instance <index-48>`
      :param config: Configuration object.
      :type config: :datamodel:`instance <index-48>`
      :return: A :py:class:`RecurrentNetwork` instance.
      :rtype: :datamodel:`instance <index-48>`

.. py:module:: parallel
   :synopsis: Runs evaluation functions in parallel subprocesses in order to evaluate multiple genomes at once.

parallel
----------
Runs evaluation functions in parallel subprocesses in order to evaluate multiple genomes at once.

  .. index:: fitness function
  .. index:: fitness

  .. py:class:: ParallelEvaluator(num_workers, eval_function, timeout=None)

    Runs evaluation functions in parallel subprocesses in order to evaluate multiple genomes at once. The analogous :py:mod:`threaded` is probably preferable
    for python implementations without a :pygloss:`GIL` (Global Interpreter Lock); note that neat-python is not currently tested vs any such implementations.

    :param int num_workers: How many workers to have in the `Pool <python:multiprocessing.pool.Pool>`.
    :param eval_function: The eval_function should take one argument - a `tuple` of (genome object, config object) - and return a single :pytypes:`float <typesnumeric>` (the genome's fitness) Note that this is not the same as how a fitness function is called by :py:meth:`Population.run <population.Population.run>`, nor by :py:class:`ThreadedEvaluator <threaded.ThreadedEvaluator>` (although it is more similar to the latter).
    :type eval_function: `function`
    :param timeout: How long (in seconds) each subprocess will be given before an exception is raised (unlimited if `None`).
    :type timeout: :pytypes:`int <typesnumeric>` or None

    .. py:method:: __del__()

       Takes care of removing the subprocesses.

    .. py:method:: evaluate(genomes, config)

      Distributes the evaluation jobs among the subprocesses, then assigns each fitness back to the appropriate genome.

      :param genomes: A list of tuples of :term:`genome_id <key>` (not used), genome.
      :type genomes: list(tuple(int, :datamodel:`instance <index-48>`))
      :param config: A `config.Config` instance.
      :type config: :datamodel:`instance <index-48>`
      
.. py:module:: population
   :synopsis: Implements the core evolution algorithm.

population
--------------
Implements the core evolution algorithm.

  .. index:: reset_on_extinction

  .. py:exception:: CompleteExtinctionException

    Raised on complete extinction (all species removed due to stagnation) unless :ref:`reset_on_extinction <reset-on-extinction-label>` is set.

  .. index:: fitness function
  .. index:: fitness
  .. index:: fitness_criterion
  .. index:: fitness_threshold
  .. index:: start_generation()
  .. index:: end_generation()
  .. index:: post_evaluate()
  .. index:: complete_extinction()
  .. index:: found_solution()
  .. index:: generation

  .. py:class:: Population(config, initial_state=None)

    This class implements the core evolution algorithm:
    1. Evaluate fitness of all genomes.
    2. Check to see if the termination criterion is satisfied; exit if it is.
    3. Generate the next :term:`generation` from the current population.
    4. Partition the new generation into species based on :term:`genetic similarity <genomic distance>`.
    5. Go to 1.

    :param config: The :py:class:`Config <config.Config>` configuration object.
    :type config: :datamodel:`instance <index-48>`
    :param initial_state: If supplied (such as by a method of the :py:class:`Checkpointer <checkpoint.Checkpointer>` class), a tuple of (``Population``, ``Species``, generation number)
    :type initial_state: None or tuple(:datamodel:`instance <index-48>`, :datamodel:`instance <index-48>`, int)
    :raises RuntimeError: If the :ref:`fitness_criterion <fitness-criterion-label>` function is invalid.

    .. index:: ! no_fitness_termination
    .. index:: ! reset_on_extinction
    .. index:: ! generation
    .. index:: ! fitness function

    .. py:method:: run(fitness_function, n=None)

      Runs NEAT's genetic algorithm for at most n generations.  If n
      is ``None``, run until a solution is found or total extinction occurs.

      The user-provided fitness_function must take only two arguments:
      1. The population as a list of (genome id, genome) tuples.
      2. The current configuration object.

      The return value of the fitness function is ignored, but it must assign
      a Python :pytypes:`float <typesnumeric>` to the ``fitness`` member of each genome.

      The fitness function is free to maintain external state, perform evaluations in :py:mod:`parallel`, etc.

      It is assumed that the fitness function does not modify the list of genomes,
      the genomes themselves (apart from updating the fitness member),
      or the configuration object.

      :param fitness_function: The fitness function to use, with arguments specified above.
      :type fitness_function: `function`
      :param n: The maximum number of generations to run (unlimited if ``None``).
      :type n: int or None
      :return: The best genome seen.
      :rtype: :datamodel:`instance <index-48>`
      :raises RuntimeError: If ``None`` for n but :ref:`no_fitness_termination <no-fitness-termination-label>` is ``True``.
      :raises CompleteExtinctionException: If all species go extinct due to `stagnation` but :ref:`reset_on_extinction <reset-on-extinction-label>` is ``False``.

      .. versionchanged:: 0.92
        :ref:`no_fitness_termination <no-fitness-termination-label>` capability added.

.. py:module:: reporting
   :synopsis: Makes possible reporter classes, which are triggered on particular events and may provide information to the user, may do something else such as checkpointing, or may do both.

reporting
-----------
Makes possible reporter classes, which are triggered on particular events and may provide information to the user, may do something else such as checkpointing, or may do both.

  .. inheritance-diagram:: reporting checkpoint.Checkpointer statistics.StatisticsReporter

  .. py:class:: ReporterSet

    Keeps track of the set of reporters and gives methods to dispatch them at appropriate points.

    .. py:method:: add(reporter)

      Adds a reporter to those to be called via :py:class:`ReporterSet` methods.

      :param reporter: A reporter instance.
      :type reporter: :datamodel:`instance <index-48>`

    .. py:method:: remove(reporter)

      Removes a reporter from those to be called via :py:class:`ReporterSet` methods.

      :param reporter: A reporter instance.
      :type reporter: :datamodel:`instance <index-48>`

    .. index:: generation

    .. py:method:: start_generation(gen)

      Calls :py:meth:`start_generation <BaseReporter.start_generation>` on each reporter in the set.

      :param int gen: The :term:`generation` number.

    .. py:method:: end_generation(config, population, species)

      Calls :py:meth:`end_generation <BaseReporter.end_generation>` on each reporter in the set.

      :param config: :py:class:`Config <config.Config>` configuration instance.
      :type config: :datamodel:`instance <index-48>`
      :param population: Current population, as a dict of unique genome :term:`ID/key <key>` vs genome.
      :type population: dict(int, :datamodel:`instance <index-48>`)
      :param species: Current species set object, such as a :py:class:`DefaultSpeciesSet <species.DefaultSpeciesSet>` instance.
      :type species: :datamodel:`instance <index-48>`

    .. py:method:: post_evaluate(config, population, species)

      Calls :py:meth:`post_evaluate <BaseReporter.post_evaluate>` on each reporter in the set.

      :param config: :py:class:`Config <config.Config>` configuration instance.
      :type config: :datamodel:`instance <index-48>`
      :param population: Current population, as a dict of unique genome :term:`ID/key <key>` vs genome.
      :type population: dict(int, :datamodel:`instance <index-48>`)
      :param species: Current species set object, such as a :py:class:`DefaultSpeciesSet <species.DefaultSpeciesSet>` instance.
      :type species: :datamodel:`instance <index-48>`
      :param best_genome: The currently highest-fitness :term:`genome`. (Ties are resolved pseudorandomly, by `dictionary <dict>` ordering.)
      :type best_genome: :datamodel:`instance <index-48>`

    .. py:method:: post_reproduction(config, population, species)

       Not currently called. Would call :py:meth:`post_reproduction <BaseReporter.post_reproduction>` on each reporter in the set.

    .. py:method:: complete_extinction()

      Calls :py:meth:`complete_extinction <BaseReporter.complete_extinction>` on each reporter in the set.

    .. py:method:: found_solution(config, generation, best)

      Calls :py:meth:`found_solution <BaseReporter.found_solution>` on each reporter in the set.

      :param config: :py:class:`Config <config.Config>` configuration instance.
      :type config: :datamodel:`instance <index-48>`
      :param int generation: Generation number.
      :param best: The currently highest-fitness :term:`genome`. (Ties are resolved pseudorandomly by `dictionary <dict>` ordering.)
      :type best: :datamodel:`instance <index-48>`

    .. py:method:: species_stagnant(sid, species)

      Calls :py:meth:`species_stagnant <BaseReporter.species_stagnant>` on each reporter in the set.

      :param int sid: The species :term:`id/key <key>`.
      :param species: The :py:class:`Species <species.Species>` instance.
      :type species: :datamodel:`instance <index-48>`

    .. py:method:: info(msg)

      Calls :py:meth:`info <BaseReporter.info>` on each reporter in the set.

      :param str msg: Message to be handled.

  .. py:class:: BaseReporter

    Abstract class defining the reporter interface expected by ReporterSet. Inheriting from it will provide a set of ``dummy`` methods to be overridden as
    desired, as follows:

    .. index:: generation

    .. py:method:: start_generation(generation)

      Called via :py:class:`ReporterSet` (by :py:meth:`population.Population.run`) at the start of each generation, prior to the invocation of the fitness function.

      :param int generation: The :term:`generation` number.

    .. index:: key

    .. py:method:: end_generation(config, population, species)

      Called via :py:class:`ReporterSet` (by :py:meth:`population.Population.run`) at the end of each :term:`generation`, after reproduction and speciation.

      :param config: :py:class:`Config <config.Config>` configuration instance.
      :type config: :datamodel:`instance <index-48>`
      :param population: Current population, as a dict of unique genome :term:`ID/key <key>` vs genome.
      :type population: dict(int, :datamodel:`instance <index-48>`)
      :param species: Current species set object, such as a :py:class:`DefaultSpeciesSet <species.DefaultSpeciesSet>` instance.
      :type species: :datamodel:`instance <index-48>`

    .. index:: fitness function

    .. py:method:: post_evaluate(config, population, species, best_genome)

      Called via :py:class:`ReporterSet` (by :py:meth:`population.Population.run`) after the fitness function is finished.

      :param config: :py:class:`Config <config.Config>` configuration instance.
      :type config: :datamodel:`instance <index-48>`
      :param population: Current population, as a dict of unique genome :term:`ID/key <key>` vs genome.
      :type population: dict(int, :datamodel:`instance <index-48>`)
      :param species: Current species set object, such as a :py:class:`DefaultSpeciesSet <species.DefaultSpeciesSet>` instance.
      :type species: :datamodel:`instance <index-48>`
      :param best_genome: The currently highest-fitness :term:`genome`. (Ties are resolved pseudorandomly, by `dictionary <dict>` ordering.)
      :type best_genome: :datamodel:`instance <index-48>`

    .. py:method:: post_reproduction(config, population, species)

      Not currently called (indirectly or directly), including by either :py:meth:`population.Population.run` or :py:class:`reproduction.DefaultReproduction`.
      Note: New members of the population likely will not have a set species.

    .. index:: reset_on_extinction

    .. py:method:: complete_extinction()

      Called via :py:class:`ReporterSet` (by :py:meth:`population.Population.run`) if complete extinction (due to stagnation) occurs, prior to
      (depending on the :ref:`reset_on_extinction <reset-on-extinction-label>` configuration setting)
      a new population being created or a :py:exc:`population.CompleteExtinctionException` being raised.

    .. index:: ! found_solution()
    .. index:: fitness_threshold
    .. index:: no_fitness_termination

    .. py:method:: found_solution(config, generation, best)

      Called via :py:class:`ReporterSet` (by :py:meth:`population.Population.run`) prior to exiting if the configured
      :ref:`fitness threshold <fitness-threshold-label>` is met, unless :ref:`no_fitness_termination <no-fitness-termination-label>` is set; if
      it is set, then called upon reaching the generation maximum - set when calling :py:meth:`population.Population.run` - and exiting for this reason.)

      :param config: :py:class:`Config <config.Config>` configuration instance.
      :type config: :datamodel:`instance <index-48>`
      :param int generation: :term:`Generation <generation>` number.
      :param best: The currently highest-fitness :term:`genome`. (Ties are resolved pseudorandomly by `dictionary <dict>` ordering.)
      :type best: :datamodel:`instance <index-48>`

      .. versionchanged:: 0.92
        :ref:`no_fitness_termination <no-fitness-termination-label>` capability added.

    .. py:method:: species_stagnant(sid, species)

      Called via :py:class:`ReporterSet` (by :py:meth:`reproduction.DefaultReproduction.reproduce`) for each species considered stagnant by the
      stagnation class (such as :py:class:`stagnation.DefaultStagnation`).

      :param int sid: The species :term:`id/key <key>`.
      :param species: The :py:class:`Species <species.Species>` instance.
      :type species: :datamodel:`instance <index-48>`

    .. py:method:: info(msg)

      Miscellaneous informational messages, from multiple parts of the library, called via :py:class:`ReporterSet`.

      :param str msg: Message to be handled.

  .. py:class:: StdOutReporter(show_species_detail)

    Uses `print` to output information about the run; an example reporter class.

    :param bool show_species_detail: Whether or not to show additional details about each species in the population.

.. py:module:: reproduction
   :synopsis: Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents.

reproduction
-----------------
Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents. For class requirements, see :ref:`reproduction-interface-label`. Implements the default NEAT-python reproduction scheme: explicit fitness sharing with fixed-time species stagnation. 

  .. py:class:: DefaultReproduction(config, reporters, stagnation)

    Implements the default NEAT-python reproduction scheme: explicit fitness sharing with fixed-time species stagnation. Inherits
    from :py:class:`config.DefaultClassConfig` the required class method :py:meth:`write_config <config.DefaultClassConfig.write_config>`.
    TODO: Provide some sort of optional cross-species performance criteria, which are then used to control stagnation and possibly the mutation
    rate configuration. This scheme should be adaptive so that species do not evolve to become "cautious" and only make very slow progress. 

    :param config: Configuration object, in this implementation a :py:class:`config.DefaultClassConfig` :datamodel:`instance <index-48>`.
    :type config: :datamodel:`instance <index-48>`
    :param reporters: A :py:class:`ReporterSet <reporting.ReporterSet>` instance.
    :type reporters: :datamodel:`instance <index-48>`
    :param stagnation: A :py:class:`DefaultStagnation <stagnation.DefaultStagnation>` instance - the current code partially depends on internals of this class (a TODO is noted to correct this).
    :type stagnation: :datamodel:`instance <index-48>`

    .. versionchanged:: 0.92
      Configuration changed to use DefaultClassConfig, instead of a dictionary, and inherit write_config.

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Provides defaults for :index:`elitism`, :index:`survival_threshold`, and :index:`min_species_size` parameters and updates
      them from the :ref:`configuration file <reproduction-config-label>`, in this implementation using :py:class:`config.DefaultClassConfig`.

      :param param_dict: Dictionary of parameters from configuration file.
      :type param_dict: dict(str, str)
      :return: Reproduction configuration object; considered opaque by rest of code, so current type returned is not required for interface.
      :rtype: DefaultClassConfig :datamodel:`instance <index-48>`

      .. versionchanged:: 0.92
        Configuration changed to use DefaultClassConfig instead of a dictionary.

    .. index:: genome

    .. py:method:: create_new(genome_type, genome_config, num_genomes)

      Required interface method. Creates ``num_genomes`` new genomes of the given type using the given configuration. Also initializes ancestry
      information (as an empty tuple).

      :param genome_type: Genome class (such as :py:class:`DefaultGenome <genome.DefaultGenome>` or :py:class:`iznn.IZGenome`) of which to create instances.
      :type genome_type: `class`
      :param genome_config: Opaque genome configuration object.
      :type genome_config: :datamodel:`instance <index-48>`
      :param int num_genomes: How many new genomes to create.
      :return: A dictionary (with the unique genome identifier as the key) of the genomes created.
      :rtype: dict(int, :datamodel:`instance <index-48>`)

    .. index:: ! pop_size
    .. index:: min_species_size

    .. py:staticmethod:: compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size)

      Apportions desired number of members per species according to fitness (adjusted by :py:meth:`reproduce` to a 0-1 scale) from out of the
      desired population size.

      :param adjusted_fitness: Mean fitness for species members, adjusted to 0-1 scale (see below).
      :type adjusted_fitness: list(:pytypes:`float <typesnumeric>`)
      :param previous_sizes: Number of members of species in population prior to reproduction.
      :type previous_sizes: list(int)
      :param int pop_size: Desired population size, as input to :py:meth:`reproduce` and :ref:`set <pop-size-label>` in the configuration file.
      :param int min_species_size: Minimum number of members per species, set via the :ref:`min_species_size <min-species-size-label>` configuration parameter (or the :ref:`elitism <elitism-label>` configuration parameter, if higher); can result in population size being above ``pop_size``.

    .. index:: pop_size
    .. index:: ! fitness function
    .. index:: ! fitness
    .. index:: key
    .. index:: ! elitism
    .. index:: ! survival_threshold
    .. index:: ! species_stagnant()
    .. index:: stagnation
    .. index:: ! info()

    .. py:method:: reproduce(config, species, pop_size, generation)

      Required interface method. Creates the population to be used in the next generation from the given configuration instance, SpeciesSet instance,
      :ref:`desired size of the population <pop-size-label>`, and current generation number.  This method is called after all genomes have been evaluated and
      their ``fitness`` member assigned.  This method should use the stagnation instance given to the initializer to remove species deemed to have stagnated.
      Note: Determines relative fitnesses by transforming into (ideally) a 0-1 scale; however, if the top and bottom fitnesses are not at least 1 apart, the
      range may be less than 0-1, as a check against dividing by a too-small number. TODO: Make minimum difference configurable (defaulting to 1 to
      preserve compatibility).

      :param config: A :py:class:`Config <config.Config>` instance.
      :type config: :datamodel:`instance <index-48>`
      :param species: A :py:class:`DefaultSpeciesSet <species.DefaultSpeciesSet>` instance. As well as depending on some of the :py:class:`DefaultStagnation <stagnation.DefaultStagnation>` internals, this method also depends on some of those of the ``DefaultSpeciesSet`` and its referenced species objects.
      :type species: :datamodel:`instance <index-48>`
      :param int pop_size: Population size desired, such as set in the :ref:`configuration file <pop-size-label>`.
      :param int generation: :term:`Generation <generation>` count.
      :return: New population, as a dict of unique genome :term:`ID/key <key>` vs :term:`genome`.
      :rtype: dict(int, :datamodel:`instance <index-48>`)

      .. versionchanged:: 0.92
        Previously, the minimum and maximum relative fitnesses were determined (contrary to the comments in the code) including members of species being removed due to
        stagnation; it is now determined using only the non-stagnant species. The minimum size of species was (and is) the greater of the
        :ref:`min_species_size <min-species-size-label>` and :ref:`elitism <elitism-label>` configuration parameters; previously, this was not taken into account for 
        :py:meth:`compute_spawn`; this made it more likely to have a population size above the :ref:`configured population size <pop-size-label>`.

.. py:module:: six_util
   :synopsis: Provides Python 2/3 portability with three dictionary iterators; copied from the `six` module.

six_util
----------
This Python 2/3 portability code was copied from the `six module <https://pythonhosted.org/six/>`_ to avoid adding it as a dependency.

  .. todo::
    Better documentation for the ``kw`` parameter in the below. Internally, these are using ``**kw`` as a **parameter** for
    keys/items/values/iterkeys/iteritems/itervalues! Is this in case someone puts in a set of key/value pairs instead of a dictionary?
    The `six documentation <https://pythonhosted.org/six/>`_ just states that this parameter is "passed to the underlying method", which is not helpful.


  .. py:function:: iterkeys(d, **kw)

    This function returns an iterator over the keys of dict d.

    :param dict d: Dictionary to iterate over
    :param kw: The function of this parameter is unclear.

  .. py:function:: iteritems(d, **kw)

    This function returns an iterator over the (key, value) pairs of dict d.

    :param dict d: Dictionary to iterate over
    :param kw: The function of this parameter is unclear.

  .. py:function:: itervalues(d, **kw)

    This function returns an iterator over the values of dict d.

    :param dict d: Dictionary to iterate over
    :param kw: The function of this parameter is unclear.

.. index:: key

.. py:module:: species
   :synopsis: Divides the population into species based on genomic distances.

species
-----------
Divides the population into species based on :term:`genomic distances <genomic distance>`.

  .. py:class:: Species(key, generation)

    Represents a :term:`species` and contains data about it such as members, fitness, and time stagnating.
    Note: :py:class:`stagnation.DefaultStagnation` manipulates many of these.

    :param int key: :term:`Identifier/key <key>`
    :param int generation: Initial :term:`generation` of appearance

    .. index:: genomic distance

    .. py:method:: update(representative, members)

      Required interface method. Updates a species instance with the current members and most-representative member (from which
      :term:`genomic distances <genomic distance>` are measured).

      :param representative: A genome instance.
      :type representative: :datamodel:`instance <index-48>`
      :param members: A `dictionary <dict>` of genome :term:`id <key>` vs genome instance.
      :type members: dict(int, :datamodel:`instance <index-48>`)

    .. py:method:: get_fitnesses()

      Required interface method (used by :py:class:`stagnation.DefaultStagnation`, for instance). Retrieves the fitnesses of each member genome.

      :return: List of fitnesses of member genomes.
      :rtype: list(:pytypes:`float <typesnumeric>`)

  .. index:: ! genomic distance

  .. py:class:: GenomeDistanceCache(config)

    Caches (indexing by :term:`genome` :term:`key`/id) :term:`genomic distance` information to avoid repeated lookups. (The
    :py:meth:`distance function <genome.DefaultGenome.distance>`, memoized by this class, is among the most time-consuming parts of the
    library, although many fitness functions are likely to far outweigh this for moderate-size populations.)

    :param config: A genome configuration instance; later used by the genome distance function.
    :type config: :datamodel:`instance <index-48>`

    .. py:method:: __call__(genome0, genome1)

      GenomeDistanceCache is called as a method with a pair of genomes to retrieve the distance.

      :param genome0: The first genome instance.
      :type genome0: :datamodel:`instance <index-48>`
      :param genome1: The second genome instance.
      :type genome1: :datamodel:`instance <index-48>`
      :return: The :term:`genomic distance`.
      :rtype: :pytypes:`float <typesnumeric>`

  .. py:class:: DefaultSpeciesSet(config, reporters)

    Encapsulates the default speciation scheme by configuring it and performing the speciation function (placing genomes into species by genetic similarity).
    :py:class:`reproduction.DefaultReproduction` currently depends on this having a ``species`` attribute consisting of a dictionary of species keys to species.
    Inherits from :py:class:`config.DefaultClassConfig` the required class method :py:meth:`write_config <config.DefaultClassConfig.write_config>`.

    :param config: A configuration object, in this implementation a :py:class:`config.Config` :datamodel:`instance <index-48>`.
    :type config: :datamodel:`instance <index-48>`
    :param reporters: A :py:class:`ReporterSet <reporting.ReporterSet>` instance giving reporters to be notified about :term:`genomic distance` statistics.
    :type reporters: :datamodel:`instance <index-48>`

    .. versionchanged:: 0.92
      Configuration changed to use DefaultClassConfig, instead of a dictionary, and inherit write_config.

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Currently, the only configuration parameter is the :ref:`compatibility_threshold <compatibility-threshold-label>`; this
      method provides a default for it and updates it from the configuration file, in this implementation using :py:class:`config.DefaultClassConfig`.

      :param param_dict: Dictionary of parameters from configuration file.
      :type param_dict: dict(str, str)
      :return: SpeciesSet configuration object; considered opaque by rest of code, so current type returned is not required for interface.
      :rtype: DefaultClassConfig :datamodel:`instance <index-48>`

      .. versionchanged:: 0.92
        Configuration changed to use DefaultClassConfig instead of a dictionary.

    .. index:: ! genomic distance
    .. index:: compatibility_threshold
    .. index:: info()

    .. py:method:: speciate(config, population, generation)

      Required interface method. Place genomes into species by genetic similarity (:term:`genomic distance`). TODO: The current code has a `docstring`
      stating that there may be a problem if all old species representatives are not dropped for each generation; it is not clear how this is consistent with the
      code in :py:meth:`reproduction.DefaultReproduction.reproduce`, such as for :ref:`elitism <elitism-label>`. TODO: Check if sorting the unspeciated
      genomes by fitness will improve speciation (by making the highest-fitness member of a species its representative).

      :param config: :py:class:`Config <config.Config>` instance.
      :type config: :datamodel:`instance <index-48>`
      :param population: Population as per the output of :py:meth:`DefaultReproduction.reproduce <reproduction.DefaultReproduction.reproduce>`.
      :type population: dict(int, :datamodel:`instance <index-48>`)
      :param int generation: Current :term:`generation` number.

    .. py:method:: get_species_id(individual_id)

      Required interface method (used by :py:class:`reporting.StdOutReporter`). Retrieves species :term:`id/key <key>` for a given genome id/key.

      :param int individual_id: Genome id/:term:`key`.
      :return: Species id/:term:`key`.
      :rtype: :pytypes:`int <typesnumeric>`

    .. py:method:: get_species(individual_id)

      Retrieves species object for a given genome :term:`id/key <key>`. May become a required interface method, and useful for some fitness
      functions already.

      :param int individual_id: Genome id/:term:`key`.
      :return: :py:class:`Species <species.Species>` containing the genome corresponding to the id/key.
      :rtype: :datamodel:`instance <index-48>`


.. index:: ! max_stagnation
.. index:: ! species_elitism

.. py:module:: stagnation
   :synopsis: Keeps track of whether species are making progress and helps remove ones that are not (for a configurable number of generations).

stagnation
--------------
Keeps track of whether species are making progress and helps remove ones that are not.

  .. index:: ! species_fitness_func
  .. index:: fitness_criterion
  .. index:: fitness_threshold
  .. index:: fitness

  .. note::

    TODO: Currently, depending on the settings for :ref:`species_fitness_func <species-fitness-func-label>` and
    :ref:`fitness_criterion <fitness-criterion-label>`, it is possible for a species with members **above** the :ref:`fitness_threshold <fitness-threshold-label>`
    level of fitness to be considered "stagnant" (including, most problematically, because they are at the limit of fitness improvement).

  .. py:class:: DefaultStagnation(config, reporters)

    Keeps track of whether species are making progress and helps remove ones that, for a
    :ref:`configurable number of generations <max-stagnation-label>`, are not. Inherits from :py:class:`config.DefaultClassConfig` the required class
    method :py:meth:`write_config <config.DefaultClassConfig.write_config>`.

    :param config: Configuration object; in this implementation, a :py:class:`config.DefaultClassConfig` instance, but should be treated as opaque outside this class.
    :type config: :datamodel:`instance <index-48>`
    :param reporters: A :py:class:`ReporterSet <reporting.ReporterSet>` instance with reporters that may need activating; not currently used.
    :type reporters: :datamodel:`instance <index-48>`

    .. versionchanged:: 0.92
      Configuration changed to use DefaultClassConfig, instead of a dictionary, and inherit write_config.

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Provides defaults for :ref:`species_fitness_func <species-fitness-func-label>`,
      :ref:`max_stagnation <max-stagnation-label>`, and :ref:`species_elitism <species-elitism-label>` parameters and updates them
      from the configuration file, in this implementation using :py:class:`config.DefaultClassConfig`.

      :param param_dict: Dictionary of parameters from configuration file.
      :type param_dict: dict(str, str)
      :return: Stagnation configuration object; considered opaque by rest of code, so current type returned is not required for interface.
      :rtype: DefaultClassConfig :datamodel:`instance <index-48>`

      .. versionchanged:: 0.92
        Configuration changed to use DefaultClassConfig instead of a dictionary.

    .. index:: fitness
    .. index:: species_elitism

    .. py:method:: update(species_set, generation)

      Required interface method. Updates species fitness history information, checking for ones that have not improved in
      :ref:`max_stagnation <max-stagnation-label>` generations, and - unless it would result in the number of species dropping below the configured
      :ref:`species_elitism <species-elitism-label>` if they were removed, in which case the highest-fitness species are spared - returns a list with
      stagnant species marked for removal. TODO: Currently interacts directly with the internals of the :py:class:`species.Species` object.
      Also, currently **both** checks for num_non_stagnant to stop marking stagnant **and** does not allow the top ``species_elitism`` species to be
      marked stagnant. While the latter could admittedly help with the problem mentioned above, the ordering of species fitness is using the
      fitness gotten from the ``species_fitness_func`` (and thus may miss high-fitness members of overall low-fitness species, depending on the
      function in use).

      :param species_set: A :py:class:`species.DefaultSpeciesSet` or compatible object.
      :type species_set: :datamodel:`instance <index-48>`
      :param int generation: The current generation.
      :return: A list of tuples of (species :term:`id/key <key>`, :py:class:`Species <species.Species>` instance, is_stagnant).
      :rtype: list(tuple(int, :datamodel:`instance <index-48>`, bool))

      .. versionchanged:: 0.92
        Species sorted (by the species fitness according to the ``species_fitness_func``) to avoid marking best-performing as stagnant even with ``species_elitism``.

.. py:module:: statistics
   :synopsis: Gathers and provides (to callers and/or to a file) information on genome and species fitness, which are the most-fit genomes, and similar.

statistics
-------------

  .. note::
    There are two design decisions to be aware of:

    * The most-fit genomes are based on the highest-fitness member of each generation; other genomes are not saved by this module (if they were, it would far worsen existing potential memory problems - see below), and it is assumed that fitnesses (as given by the :index:`fitness function <single: fitness function>`) are not relative to others in the generation (also assumed by the use of the :ref:`fitness threshold <fitness-threshold-label>` as a signal for exiting). Code violating this assumption (e.g., with competitive coevolution) will need to use different statistical gathering methods.
    * Generally reports or records a per-generation list of values; the numeric position in the list may not correspond to the generation number if there has been a restart, such as via the :py:mod:`checkpoint` module.

    There is also a TODO item: Currently keeps accumulating information in memory, which may be a problem in long runs.

  .. py:class:: StatisticsReporter(BaseReporter)

    Gathers (via the reporting interface) and provides (to callers and/or to a file) the most-fit genomes and information on genome and species fitness
    and species sizes. 

    .. py:method:: post_evaluate(config, population, species, best_genome)

      Called as part of the :py:class:`reporting.BaseReporter` interface after the evaluation at the start of each generation;
      see :py:meth:`BaseReporter.post_evaluate <reporting.BaseReporter.post_evaluate>`.
      Information gathered includes a copy of the best genome in each generation and the fitnesses of each member of each species.

    .. py:method:: get_fitness_stat(f)

      Calls the given function on the genome fitness data from each recorded generation and returns the resulting list.

      :param f: A function that takes a list of scores and returns a summary statistic (or, by returning a list or tuple, multiple statistics) such as ``mean`` or ``stdev``.
      :type f: `function`
      :return: A list of the results from function f for each generation.
      :rtype: list

    .. py:method:: get_fitness_mean()

      Gets the per-generation mean fitness. A wrapper for :py:meth:`get_fitness_stat` with the function being ``mean``.

      :return: List of mean genome fitnesses for each generation.
      :rtype: list(:pytypes:`float <typesnumeric>`)

    .. py:method:: get_fitness_median()

      Gets the per-generation median fitness. A wrapper for :py:meth:`get_fitness_stat` with the function being `median2`. Not currently used internally.

      .. versionadded:: 0.92

    .. py:method:: get_fitness_stdev()

      Gets the per-generation standard deviation of the fitness. A wrapper for :py:meth:`get_fitness_stat` with the function being ``stdev``.

      :return: List of standard deviations of genome fitnesses for each generation.
      :rtype: list(:pytypes:`float <typesnumeric>`)

    .. py:method:: best_unique_genomes(n)

      Returns the ``n`` most-fit genomes, with no duplication (from the most-fit genome passing unaltered to the next generation), sorted in decreasing
      fitness order.

      :param int n: Number of most-fit genomes to return.
      :return: List of ``n`` most-fit genomes (as genome instances).
      :rtype: list(:datamodel:`instance <index-48>`)

    .. py:method:: best_genomes(n)

      Returns the ``n`` most-fit genomes, possibly with duplicates, sorted in decreasing fitness order.

      :param int n: Number of most-fit genomes to return.
      :return: List of ``n`` most-fit genomes (as genome instances).
      :rtype: list(:datamodel:`instance <index-48>`)

    .. py:method:: best_genome()

      Returns the most-fit genome ever seen. A wrapper around :py:meth:`best_genomes`.

      :return: The most-fit genome.
      :rtype: :datamodel:`instance <index-48>`

    .. py:method:: get_species_sizes()

      Returns a by-generation list of lists of species sizes. Note that some values may be 0, if a species has either not yet been seen or has been
      removed due to :py:mod:`stagnation`; species without generational overlap may be more similar in :term:`genomic distance` than the configured
      :ref:`compatibility_threshold <compatibility-threshold-label>` would otherwise allow.

      :return: List of lists of species sizes, ordered by species :term:`id/key <key>`.
      :rtype: list(list(int))

    .. py:method:: get_species_fitness(null_value='')

      Returns a by-generation list of lists of species fitnesses; the fitness of a species is determined by the ``mean`` fitness of the genomes in the species,
      as with the reproduction distribution by :py:class:`reproduction.DefaultReproduction`. The ``null_value`` parameter is used for species not present in a
      particular generation (see :py:meth:`above <get_species_sizes>`).

      :param str null_value: What to put in the list if the species is not present in a particular generation.
      :return: List of lists of species fitnesses, ordered by species :term:`id/key <key>`.
      :rtype: list(list(float or str))

    .. py:method:: save_genome_fitness(delimiter=' ', filename='fitness_history.csv', with_cross_validation=False)

      Saves the population's best and mean fitness (using the `csv` package). At some point in the future, cross-validation fitness may be usable (via, for
      instance, the fitness function using alternative test situations/opponents and recording this in a ``cross_fitness`` attribute; this can be used for, e.g.,
      preventing overfitting); currently, ``with_cross_validation`` should always be left at its ``False`` default.

      :param str delimiter: Delimiter between columns in the file; note that the default is not ',' as may be otherwise implied by the ``csv`` file extension (which refers to the package used).
      :param str filename: The filename to open (for writing, not appending) and write to.
      :param bool with_cross_validation: For future use; currently, leave at its ``False`` default.

    .. py:method:: save_species_count(delimiter=' ', filename='speciation.csv')

      Logs speciation throughout evolution, by tracking the number of genomes in each species. Uses :py:meth:`get_species_sizes`; see that method for
      more information.

      :param str delimiter: Delimiter between columns in the file; note that the default is not ',' as may be otherwise implied by the ``csv`` file extension (which refers to the `csv` package used).
      :param str filename: The filename to open (for writing, not appending) and write to.

    .. py:method:: save_species_fitness(delimiter=' ', null_value='NA', filename='species_fitness.csv')

      Logs species' mean fitness throughout evolution. Uses :py:meth:`get_species_fitness`; see that method for more information on, for
      instance, ``null_value``.

      :param str delimiter: Delimiter between columns in the file; note that the default is not ',' as may be otherwise implied by the ``csv`` file extension (which refers to the `csv` package used).
      :param str null_value: See :py:meth:`get_species_fitness`.
      :param str filename: The filename to open (for writing, not appending) and write to.

    .. py:method:: save()

      A wrapper for :py:meth:`save_genome_fitness`, :py:meth:`save_species_count`, and :py:meth:`save_species_fitness`;
      uses the default values for all three.

.. py:module:: threaded
   :synopsis: Runs evaluation functions in parallel threads in order to evaluate multiple genomes at once.

threaded
----------
Runs evaluation functions in parallel threads (using the python library module `threading <https://docs.python.org/3.5/library/threading.html>`_) in order to evaluate multiple genomes at once. Probably preferable to :py:mod:`parallel` for python implementations without a :pygloss:`GIL` (Global Interpreter Lock); note, however, that neat-python is not currently tested on any such implementation.

  .. index:: fitness function
  .. index:: fitness

  .. py:class:: ThreadedEvaluator(num_workers, eval_function)

    Runs evaluation functions in parallel threads in order to evaluate multiple genomes at once.

    :param int num_workers: How many worker threads to use.
    :param eval_function: The eval_function should take two arguments - a genome object and a config object - and return a single :pytypes:`float <typesnumeric>` (the genome's fitness) Note that this is not the same as how a fitness function is called by :py:meth:`Population.run <population.Population.run>`, nor by :py:class:`ParallelEvaluator <parallel.ParallelEvaluator>` (although it is more similar to the latter).
    :type eval_function: `function`

    .. py:method:: __del__()

      Attempts to take care of removing each worker thread, but deliberately calling ``self.stop()`` in the threads may be needed.
      TODO: Avoid reference cycles to ensure this method is called. (Perhaps use `weakref`, depending on what the cycles are?
      Note that weakref is not compatible with saving via `pickle`, so all of them will need to be removed prior to any save.)

    .. py:method:: start()

      Starts the worker threads, if in the primary thread.

    .. py:method:: stop()

      Stops the worker threads and waits for them to finish.

    .. py:method:: _worker():

      The worker function.

    .. py:method:: evaluate(genomes, config)

      Starts the worker threads if need be, queues the evaluation jobs for the worker threads, then assigns each fitness back to the appropriate genome.

      :param genomes: A list of tuples of :term:`genome_id <key>`, genome instances.
      :type genomes: list(tuple(int, :datamodel:`instance <index-48>`))
      :param config: A `config.Config` instance.
      :type config: :datamodel:`instance <index-48>`

  .. versionadded:: 0.92

:ref:`Table of Contents <toc-label>`
