
Module summaries
==================

.. default-role:: any

.. todo::

  Finish putting in all needed material from modules; add links; go over parameters as used in code to make sure are described correctly.

.. index:: activation function

.. py:module:: activations
   :synopsis: Has the built-in activation functions and code for using them and adding new user-defined ones.

activations
---------------

  .. py:exception:: InvalidActivationFunction(Exception)

    Exception called if an activation function being added is invalid according to the `validate_activation` function.

  .. py:function:: validate_activation(function)

    Checks to make sure its parameter is a function that takes a single argument. TODO: Currently raises a deprecation warning due to changes in `inspect`.

    :param object function: Object to be checked.
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
      :raises InvalidActivationFunction: If the function is not known.

    .. py:method:: is_valid(name)

      Checks whether the named function is a known activation function.

      :param str name: The name of the function.
      :return: Whether or not the function is known.
      :rtype: bool

.. note::
  TODO: Suggested simplification for the below: Make ``__config_items__`` a list of lists/tuples, with the latter containing (name, value_type, default) -
  no default if the last is None. This would also allow moving get_config_params into the BaseAttribute class, although config_item_names may require
  some modifications. (A default capability will be needed for future expansions of the attributes, such as different types of initializations.)

.. py:module:: attributes
   :synopsis: Deals with attributes used by genes.

attributes
-------------

  .. inheritance-diagram:: attributes

  .. py:class:: BaseAttribute(name)

    Superclass for the type-specialized attribute subclasses, used by genes (such as via the :py:class:`genes.BaseGene` implementation). Calls
    `config_item_names` to set up a listing of the names of configuration items using `setattr`.

    .. py:method:: config_item_names()

      Formats a list of configuration item names by combining the attribute's name with the attribute class' list of ``__config_items__``.

      :return: A list of configuration item names.
      :rtype: list(str)

  .. py:class:: FloatAttribute(BaseAttribute)

    Class for numeric :term:`attributes` such as the :term:`response` of a :term:`node`; includes code for configuration, creation, and mutation.

    .. py:method:: get_config_params()

      Uses `config_item_names` to get its list of configuration item names, then gets a `float`-type :py:class:`config.ConfigParameter` instance for each
      and returns it.

      :return: A list of ``ConfigParameter`` instances.
      :rtype: list(object)

    .. py:method:: clamp(value, config)

      Gets the minimum and maximum values desired from ``config``, then ensures that the value is between them.

      :param float value: The value to be clamped.
      :param object config: The configuration object from which the minimum and maximum desired values are to be retrieved.
      :return: The value, if it is within the desired range, or the appropriate end of the range, if it is not.
      :rtype: float

    .. py:method:: init_value(config)

      Initializes the attribute's value, (currently always) using a gaussian distribution with the configured mean and standard deviation followed by `clamp` to
      keep the result within the desired range.

      :param object config: The configuration object from which the mean and standard deviation values are to be retrieved.
      :return: The new value.
      :rtype: float

    .. index:: ! mutation

    .. py:method:: mutate_value(value, config)

      May replace (as if reinitializing, using `init_value`), mutate (using a 0-mean gaussian distribution with a configured standard
      deviation from ``mutate_power``), or leave alone the input value, depending on the configuration settings (of ``replace_rate`` and ``mutate_rate``).
      TODO: Why check vs `random` if the ``replace_rate`` and ``mutate_rate`` are 0? Also note that the ``replace_rate`` is likely to be lower, so should
      be checked second.

      :param float value: The current value of the attribute.
      :param object config: The configuration object from which the parameters are to be extracted.
      :return: Either the original value, if unchanged, or the new value.
      :rtype: float

  .. py:class:: BoolAttribute(BaseAttribute)

    Class for boolean :term:`attributes` such as whether a :term:`connection` is :term:`enabled` or not; includes code for configuration, creation, and mutation.

    .. py:method:: get_config_params()

      Uses `config_item_names` to get its list of configuration item names, then gets a `bool`-type or `float`-type :py:class:`config.ConfigParameter`
      instance for each and returns it.

      :return: A list of ``ConfigParameter`` instances.
      :rtype: list(object)

    .. py:method:: init_value(config)

      Initializes the attribute's value, either using a configured default or (if the default is ``None``) with a 50/50 chance of ``True`` or ``False``.

      :param object config: The configuration object from which the default parameter is to be retrieved.
      :return: The new value.
      :rtype: bool

    .. index:: ! mutation

    .. py:method:: mutate_value(value, config)

      With a frequency determined by the ``mutate_rate`` (which is more precisely a ``replace_rate``) configuration parameter, replaces
      the value with a 50/50 chance of ``True`` or ``False``; note that this has a 50% chance of leaving the value unchanged. TODO: Have different
      chances possible of :term:`mutation` in each direction. Also, do not check vs `random` if the ``mutate_rate`` is 0.

      :param bool value: The current value of the attribute.
      :param object config: The configuration object from which the ``mutate_rate`` parameter is to be extracted.
      :return: Either the original value, if unchanged, or the new value.
      :rtype: bool

  .. py:class:: StringAttribute(BaseAttribute)

    Class for string attributes such as the :term:`aggregation function` of a :term:`node`, which are selected from a list of options;
    includes code for configuration, creation, and mutation.

    .. py:method:: get_config_params()

      Uses `config_item_names` to get its list of configuration item names, then gets a `str`-type, `list`-type or `float`-type :py:class:`config.ConfigParameter`
      instance for each and returns it.

      :return: A list of ``ConfigParameter`` instances.
      :rtype: list(object)

    .. py:method:: init_value(config)

      Initializes the attribute's value, either using a configured default or (if the default is either ``None`` or ``random``) with a randomly-chosen member
      of the ``options`` (each having an equal chance). Note: It is possible for the default value, if specifically configured, to **not** be one of the options.

      :param object config: The configuration object from which the default and, if necessary, ``options`` parameters are to be retrieved.
      :return: The new value.
      :rtype: str

    .. index:: ! mutation

    .. py:method:: mutate_value(value, config)

      With a frequency determined by the ``mutate_rate`` (which is more precisely a ``replace_rate``) configuration parameter, replaces
      the value with an one of the ``options``, with each having an equal chance; note that this can be the same value as before. (It is possible to crudely
      alter the chances of what is chosen by listing a given option more than once, although this is inefficient given the use of the `random.choice` function.)
      TODO: Do not check vs `random` if the ``mutate_rate`` is 0. (Longer-term, add configurable probabilities of which option is used; eventually, as with the
      improved version of RBF-NEAT, separate genes for the likelihoods of each (but always doing some change, to prevent overly-conservative evolution
      due to its inherent short-sightedness), allowing the genomes to control the distribution of options, will be desirable.)

.. py:module:: checkpoint
   :synopsis: Uses `pickle` to save and restore populations (and other aspects of the simulation state).

checkpoint
---------------

  .. py:class:: Checkpointer(generation_interval=100, time_interval_seconds=300)

    A reporter class that performs checkpointing using :py:mod:`pickle` to save and restore populations (and other aspects of the simulation state). It saves
    the current state every ``generation_interval`` generations or ``time_interval_seconds`` seconds, whichever happens first.
    Subclasses :py:class:`reporting.BaseReporter`. (The potential save point is at the end of a generation.)

    :param generation_interval: If not None, maximum number of generations between checkpoints.
    :type generation_interval: int or None
    :param time_interval_seconds: If not None, maximum number of seconds between checkpoints.
    :type time_interval_seconds: float or None

    .. py:staticmethod:: save_checkpoint(config, population, species, generation)

      Saves the current simulation (including randomization) state to :file:`neat-checkpoint-{generation}`, with ``generation`` being the generation number.

    .. py:staticmethod:: restore_checkpoint(filename)

      Resumes the simulation from a previous saved point. Loads the specified file, sets the randomization state, and returns
      a :py:class:`population.Population` object set up with the rest of the previous state.

      :param str filename: The file to be restored from.
      :return: Object that can be used with :py:meth:`Population.run <population.Population.run>` to restart the simulation.
      :rtype: :py:class:`Population <population.Population>` object.

.. index:: fitness_criterion
.. index:: fitness_threshold
.. index:: pop_size
.. index:: reset_on_extinction

.. py:module:: config
   :synopsis: Does general configuration parsing; used by other classes for their configuration.

config
--------

  .. py:class:: ConfigParameter(name, value_type)

    Does initial handling of a particular configuration parameter.

    :param str name: The name of the configuration parameter.
    :param str value_type: The type that the configuration parameter should be; must be one of ``str``, ``int``, ``bool``, ``float``, or ``list``.

    .. py:method:: __repr__()

      Returns a representation of the class suitable for use in code for initialization.

      :return: Representation as for `repr`.
      :rtype: str

    .. py:method:: parse(section, config_parser)

      Uses the supplied configuration parser (either from the :py:class:`configparser.ConfigParser` class, or - for 2.7 - the
      `ConfigParser.SafeConfigParser class <https://docs.python.org/2.7/library/configparser.html#ConfigParser.SafeConfigParser>`_) to gather the
      configuration parameter from the appropriate configuration file :ref:`section <configuration-file-sections-label>`. Parsing varies depending on the type.

      :param str section: The section name, taken from the `__name__` attribute of the class to be configured (or ``NEAT`` for those parameters).
      :param object config_parser: The configuration parser to be used.
      :return: The configuration parameter value, in stringified form unless a list.
      :rtype: str or list

    .. py:method:: interpret(config_dict)

      Takes a `dictionary <dict>` of configuration parameters, as output by the configuration parser called in :py:meth:`parse`, and interprets them into the
      proper type, with some error-checking.

      :param dict config_dict: Configuration parameters as output by the configuration parser.
      :return: The configuration parameter value
      :rtype: str or int or bool or float or list

    .. py:method:: format(value)

      Depending on the type of configuration parameter, returns either a space-separated list version, for ``list``  parameters, or the stringified version
      (using `str`), of ``value``.

      :param value: Configuration parameter value to be formatted.
      :type value: str or int or bool or float or list

  .. py:function:: write_pretty_params(f, config, params)

    Prints configuration parameters, with justification based on the longest configuration parameter name.

    :param f: `File object <file>` to be written to.
    :type f: `file`
    :param object config: Configuration object from which parameter values are to be fetched (using `getattr`).
    :param list params: List of :py:class:`ConfigParameter` instances giving the names of interest and the types of parameters.

  .. py:class:: Config(genome_type, reproduction_type, species_set_type, stagnation_type, filename)

    A simple container for user-configurable parameters of NEAT. The four parameters ending in ``_type`` may be the built-in ones or user-provided objects,
    which must make available the methods ``parse_config`` and ``write_config``, plus others depending on which object it is. (For more information on the
    objects, see below and :ref:`customization-label`.) ``Config`` itself takes care of the ``NEAT`` parameters. For a description of the configuration file,
    see :ref:`configuration-file-description-label`.

    :param object genome_type: Specifies the genome class used, such as :py:class:`genome.DefaultGenome` or :py:class:`iznn.IZGenome`. See :ref:`genome-interface-label` for the needed interface.
    :param object reproduction_type: Specifies the reproduction class used, such as :py:class:`reproduction.DefaultReproduction`. See :ref:`reproduction-interface-label` for the needed interface.
    :param object species_set_type: Specifies the species set class used, such as :py:class:`species.DefaultSpeciesSet`.
    :param object stagnation_type: Specifies the stagnation class used, such as :py:class:`stagnation.DefaultStagnation`.
    :param str filename: Pathname for configuration file to be opened, read, processed by a parser from the :py:class:`configparser.ConfigParser` class (or, for 2.7, the `ConfigParser.SafeConfigParser class <https://docs.python.org/2.7/library/configparser.html#ConfigParser.SafeConfigParser>`_), the ``NEAT`` section handled by ``Config``, and then other sections passed to the ``parse_config`` methods of the appropriate classes.
    :raises AssertionError: If any of the objects lack a ``parse_config`` method.

    .. py:method:: save(filename)

      Opens the specified file for writing (not appending) and outputs a configuration file from the current configuration. Uses :py:func:`write_pretty_params` for
      the ``NEAT`` parameters and the appropriate class ``write_config`` methods for the other sections.

      :param str filename: The configuration file to be written.

.. py:module:: ctrnn
   :synopsis: Handles the continuous-time recurrent neural network implementation.

ctrnn
-------

  .. py:class:: CTRNNNodeEval(time_constant, activation, aggregation, bias, response, links)

    Sets up the basic :doc:`ctrnn` (:term:`continuous-time` :term:`recurrent` neural network) :term:`nodes <node>`.

    :param float time_constant: Controls how fast the node responds; :math:`\tau_i` from :doc:`ctrnn`.
    :param activation: :term:`Activation function <activation function>` for the node.
    :type activation: `function`
    :param aggregation: :term:`Aggregation function <aggregation function>` for the node.
    :type aggregation: `function`
    :param float bias: :term:`Bias <bias>` for the node.
    :param float response: :term:`Response <response>` multiplier for the node.
    :param links: List of other nodes providing input, as tuples of (input :term:`key`, :term:`weight`)
    :type links: list(tuple(int,float))

  .. py:class:: CTRNN(inputs, outputs, node_evals)

    Sets up the :doc:`ctrnn` network itself.

    .. index:: recurrent

    .. py:method:: reset()

      Resets the time and all node activations to 0 (necessary due to otherwise retaining state via :term:`recurrent` connections).

    .. index:: ! continuous-time

    .. py:method:: advance(inputs, advance_time, time_step=None)

      Advance the simulation by the given amount of time, assuming that inputs are
      constant at the given values during the simulated time.

      :param list inputs: The values for the :term:`input nodes <input node>`.
      :param float advance_time: How much time to advance the network before returning the resulting outputs.
      :param float time_step: How much time per step to advance the network; the default of ``None`` will currently result in an error, but it is planned to determine it automatically.
      :return: The values for the :term:`output nodes <output node>`.
      :rtype: list
      :raises NotImplementedError: If a ``time_step`` is not given.

    .. py:staticmethod:: create(genome, config, time_constant)

      Receives a genome and returns its phenotype (a :py:class:`CTRNN` with :py:class:`CTRNNNodeEval` :term:`nodes <node>`).

      :param object genome: A :py:class:`genome.DefaultGenome` instance.
      :param object config: A :py:class:`config.Config` instance.
      :param float time_constant: Used for the :py:class:`CTRNNNodeEval` initializations.

.. py:module:: genes
   :synopsis: Handles node and connection genes.

genes
--------

  .. inheritance-diagram:: genes iznn.IZNodeGene

  .. index:: key
  .. index:: ! gene

  .. py:class:: BaseGene(key)

    Handles functions shared by multiple types of genes (both :term:`node` and :term:`connection`), including :term:`crossover` and
    calling :term:`mutation` methods.

    :param int key: The gene :term:`identifier <key>`. Note: For connection genes, determining whether they are :term:`homologous` (for :term:`genomic distance` and :term:`crossover` determination) uses the identifiers of the connected nodes, not the connection gene's identifier.

    .. py:method:: __str__()

      Converts gene attributes into a printable format.

      :return: Stringified gene instance.
      :rtype: str

    .. py:method:: __lt__(other)

      Allows sorting genes by :term:`keys <key>`.

      :param object other: The other `BaseGene` object.
      :return: Whether the calling instance's key is less than that of the ``other`` instance.
      :rtype: bool

    .. py:classmethod:: parse_config(config, param_dict)

      Placeholder; parameters are entirely in gene :term:`attributes`.

    .. py:classmethod:: get_config_params()

      Fetches configuration parameters from each gene class' ``__gene_attributes__`` list (using
      :py:meth:`FloatAttribute.get_config_params <attributes.FloatAttribute.get_config_params>`,
      :py:meth:`BoolAttribute.get_config_params <attributes.BoolAttribute.get_config_params>`,
      or :py:meth:`StringAttribute.get_config_params <attributes.StringAttribute.get_config_params>` as appropriate for each listed attribute).
      Used by :py:class:`genome.DefaultGenomeConfig` to include gene parameters in its configuration parameters.

      :return: List of configuration parameters (as :py:class:`config.ConfigParameter` instances) for the gene attributes.
      :rtype: list(object)

    .. py:method:: init_attributes(config)

      Initializes its gene attributes using the supplied configuration object and :py:meth:`FloatAttribute.init_value <attributes.FloatAttribute.init_value>`,
      :py:meth:`BoolAttribute.init_value <attributes.BoolAttribute.init_value>`, or
      :py:meth:`StringAttribute.init_value <attributes.StringAttribute.init_value>` as appropriate.

      :param object config: Configuration object to be used by the appropriate :py:mod:`attributes` class.

    .. index::
      see: mutate; mutation
    .. index:: ! mutation

    .. py:method:: mutate(config)

      :term:`Mutates <mutation>` (possibly) its gene attributes using the supplied configuration object and
      :py:meth:`FloatAttribute.init_value <attributes.FloatAttribute.mutate_value>`,
      :py:meth:`BoolAttribute.init_value <attributes.BoolAttribute.mutate_value>`, or
      :py:meth:`StringAttribute.init_value <attributes.StringAttribute.mutate_value>` as appropriate.

      :param object config: Configuration object to be used by the appropriate :py:mod:`attributes` class.

    .. py:method:: copy()

      Makes a copy of itself, including its subclass, :term:`key`, and all gene attributes.

      :return: A copied gene
      :rtype: object

    .. index:: ! crossover

    .. py:method:: crossover(gene2)

      Creates a new gene via :term:`crossover` - randomly inheriting attributes from its parents. The two genes must be :term:`homologous`, having
      the same :term:`key`/id.

      :param object gene2: The other gene.
      :return: A new gene, with the same key/id, with other attributes being copied randomly (50/50 chance) from each parent gene.
      :rtype: object

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

      :param object other: The other ``DefaultNodeGene``.
      :param object config: The genome configuration object.
      :return: The contribution of this pair to the :term:`genomic distance` between the source genomes.
      :rtype: float

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

      :param object other: The other ``DefaultConnectionGene``.
      :param object config: The genome configuration object.
      :return: The contribution of this pair to the :term:`genomic distance` between the source genomes.
      :rtype: float

.. todo::

   Explain more regarding parameters of the below; add all methods!

.. py:module:: genome
   :synopsis: Handles genomes (individuals in the population).

genome
-----------

  .. inheritance-diagram:: genome iznn.IZGenome

  .. py:function:: product(x)

    Used to implement a product (:math:`\[\prod x\]`) :term:`aggregation function`.

    :param x: The inputs to be multiplied together.
    :type x: list(float)

  .. index:: ! aggregation function
  .. index:: initial_connection

  .. py:class:: DefaultGenomeConfig(params)

    Does the configuration for the DefaultGenome class. Has the `dictionary <dict>` ``aggregation_function_defs``, which
    defines the available :term:`aggregation functions <aggregation function>`, and the `list <list>` ``allowed_connectivity``, which defines the available
    values for :ref:`initial_connection <initial-connection-config-label>`. Includes parameters taken from the configured gene classes, such
    as :py:class:`genes.DefaultNodeGene`, :py:class:`genes.DefaultConnectionGene`, or :py:class:`iznn.IZNodeGene`.

    :param dict params: Parameters from configuration file and DefaultGenome initialization (by parse_config).

    .. index:: ! activation function

    .. py:method:: add_activation(name, func)

      Adds a new :term:`activation function`, as described in :ref:`customization-label`.
      Uses :py:meth:`ActivationFunctionSet.add <activations.ActivationFunctionSet.add>`.

      :param str name: The name by which the function is to be known in the :ref:`configuration file <activation-function-config-label>`.
      :param func: A function meeting the requirements of :py:func:`activations.validate_activation`.
      :type func: `function`

    .. py:method:: save(f)

      Saves the :ref:`initial_connection <initial-connection-config-label>` configuration and uses :py:func:`config.write_pretty_params` to write out the
      other parameters.

      :param f: The `File object <file>` to be written to.
      :type f: `file`

    .. index:: ! key

    .. py:method:: get_new_node_key(node_dict)

      Finds the next unused node :term:`key`.

      :param dict node_dict: A dictionary of node keys vs nodes
      :return: A currently-unused node key.
      :rtype: int

  .. index:: key

  .. py:class:: DefaultGenome(key)

    A :term:`genome` for generalized neural networks. For class requirements, see :ref:`genome-interface-label`.
    Terminology:
    pin - Point at which the network is conceptually connected to the external world; pins are either input or output.
    node - Analog of a physical neuron.
    connection - Connection between a pin/node output and a node's input, or between a node's output and a pin/node input.
    key - Identifier for an object, unique within the set of similar objects.
    Design assumptions and conventions.
    1. Each output pin is connected only to the output of its own unique neuron by an implicit connection with weight one. This connection is permanently enabled.
    2. The output pin's key is always the same as the key for its associated neuron.
    3. Output neurons can be modified but not deleted.
    4. The input values are applied to the input pins unmodified.

    :param int key: :term:`Identifier <key>` for this individual/genome.

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Provides default :term:`node` and :term:`connection` :term:`gene` specifications (from :py:mod:`genes`) and
      uses `DefaultGenomeConfig` to do the rest of the configuration.

      :param dict param_dict: Dictionary of parameters from configuration file.
      :return: Configuration object; considered opaque by rest of code, so type may vary by implementation (here, a `DefaultGenomeConfig` instance).
      :rtype: object

    .. py:classmethod:: write_config(f, config)

      Required interface method. Saves configuration using :py:meth:`DefaultGenomeConfig.save`.

      :param f: `File object <file>` to write to.
      :type f: `file`
      :param object config: Configuration object (here, a `DefaultGenomeConfig` instance).

    .. index:: ! initial_connection
    .. index:: hidden node
    .. index:: input node
    .. index:: output node

    .. py:method:: configure_new(config)

      Required interface method. Configures a new genome (itself) based on the given
      configuration object, including genes for :term:`connectivity <connection>` (based on :ref:`initial_connection <initial-connection-config-label>`) and
      starting :term:`nodes <node>` (as defined by :term:`num_hidden <hidden node>`, :term:`num_inputs <input node>`, and
      :term:`num_outputs <output node>` in the :ref:`configuration file <num-nodes-config-label>`.

      :param object config: Genome configuration object.

    .. index:: ! crossover

    .. py:method:: configure_crossover(genome1, genome2, config)

      Required interface method. Configures a new genome (itself) by :term:`crossover` from two parent genomes. :term:`disjoint`
      or :term:`excess` genes are inherited from the fitter of the two parents, while :term:`homologous` genes use the gene class' crossover function
      (e.g., :py:meth:`genes.BaseGene.crossover`).

      :param object genome1: The first parent genome.
      :param object genome2: The second parent genome.
      :param object config: Genome configuration object.

    .. index:: ! mutation

    .. py:method:: mutate(config)

      Required interface method. :term:`Mutates <mutation>` this genome. What mutations take place are determined by configuration file settings, such
      as :ref:`node_add_prob <node-add-prob-label>` and ``node_delete_prob`` for the likelihood of adding or removing a :term:`node` and
      :ref:`conn_add_prob <conn-add-prob-label>` and ``conn_delete_prob`` for the likelihood of adding or removing a :term:`connection`. (Currently,
      more than one of these can happen with a call to ``mutate``; a TODO is to add a configuration item to choose whether or not multiple mutations
      can happen simultaneously.) Non-structural mutations (to gene :term:`attributes`) are performed by calling the appropriate ``mutate`` method(s) for
      connection and node genes (generally :py:meth:`genes.BaseGene.mutate`).

      :param object config: Genome configuration object.

    .. index:: node

    .. py:method:: mutate_add_node(config)

      Takes a randomly-selected existing connection, turns its :term:`enabled` attribute to ``False``, and makes two new (enabled) connections with a
      new :term:`node` between them, which join the now-disabled connection's nodes. The connection weights are chosen so as to potentially have
      roughly the same behavior as the original connection, although this will depend on the :term:`activation function`, :term:`bias`, and
      :term:`response` multiplier of the new node. TODO: Particularly if the configuration is changed to only allow one structural mutation, then if there
      are no connections, call :py:meth:`mutate_add_connection` instead of returning.

      :param object config: Genome configuration object.

    .. index:: ! connection

    .. py:method:: add_connection(config, input_key, output_key, weight, enabled)

      Adds a specified new connection; its :term:`key` is the `tuple` of ``(input_key, output_key)``. TODO: Add validation of this connection addition.

      :param object config: Genome configuration object
      :param int input_key: :term:`Key <key>` of the input node.
      :param int output_key: Key of the output node.
      :param float weight: The :term:`weight` the new connection should have.
      :param bool enabled: The :term:`enabled` attribute the new connection should have.

    .. index:: ! feed_forward
    .. index:: connection

    .. py:method:: mutate_add_connection(config)

      Attempts to add a randomly-selected new connection, with some filtering:
      1. :term:`input nodes <input node>` cannot be at the output end.
      2. Existing connections cannot be duplicated. TODO: If a selected existing connection is not :term:`enabled`, have some configurable chance that it will become enabled.
      3. Two :term:`output nodes <output node>` cannot be connected together.
      4. If :ref:`feed_forward <feed-forward-config-label>` is set to ``True`` in the configuration file, connections cannot create :py:func:`cycles <graphs.creates_cycle>`.

      :param object config: Genome configuration object

    .. py:method:: mutate_delete_node(config)

      Deletes a randomly-chosen (non-:term:`output <output node>`/input) node along with its connections.

      :param object config: Genome configuration object

    .. py:method:: mutate_delete_connection()

      Deletes a randomly-chosen connection. TODO: If the connection is :term:`enabled`, have an option to - possibly with a :term:`weight`-dependent
      chance - turn its enabled attribute to ``False`` instead.

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

      :param object other: The other DefaultGenome instance (genome) to be compared to.
      :param object config: The genome configuration object.
      :return: The genomic distance.
      :rtype: float

    .. py:method:: size()

      Required interface method. Returns genome ``complexity``, taken to be (number of nodes, number of enabled connections); currently only used
      for reporters - some retrieve this information for the highest-fitness genome at the end of each generation.

    .. py:method:: __str__()

      Gives a listing of the genome's nodes and connections.

      :return: Node and connection information.
      :rtype: str

    .. index:: node

    .. py:staticmethod:: create_node(config, node_id)

      Creates a new node with the specified :term:`id <key>` (including for its :term:`gene`), using the specified configuration object to retrieve the proper
      node gene type and how to initialize its attributes.

      :param object config: The genome configuration object.
      :param int node_id: The key for the new node.
      :return: The new node object.
      :rtype: object

    .. index:: connection

    .. py:staticmethod:: create_connection(config, input_id, output_id)

      Creates a new connection with the specified :term:`id <key>` pair as its key (including for its :term:`gene`, as a `tuple`), using the specified
      configuration object to retrieve the proper connection gene type and how to initialize its attributes.

      :param object config: The genome configuration object.
      :param int input_id: The input end's key.
      :param int output_id: The output end's key.
      :return: The new connection object.
      :rtype: object

    .. index:: ! initial_connection

    .. py:method:: connect_fs_neat_nohidden(config)

      Connect one randomly-chosen input to all :term:`output nodes <output node>` (FS-NEAT without connections to :term:`hidden nodes <hidden node>`,
      if any). Previously called ``connect_fs_neat``. Implements the ``fs_neat_nohidden`` setting for :ref:`initial_connection <initial-connection-config-label>`.

      :param object config: The genome configuration object.

    .. py:method:: connect_fs_neat_hidden(config)

      Connect one randomly-chosen input to all :term:`hidden nodes <hidden node>` and :term:`output nodes <output node>` (FS-NEAT with
      connections to hidden nodes, if any). Implements the ``fs_neat_hidden`` setting for :ref:`initial_connection <initial-connection-config-label>`.

      :param object config: The genome configuration object.


.. index:: feed_forward
.. index:: feedforward
.. index::
  see: feed-forward; feedforward
.. index:: recurrent

.. py:module:: graphs
   :synopsis: Directed graph algorithm implementations.

graphs
---------

  .. py:function:: creates_cycle(connections, test)

    Returns true if the addition of the ``test`` :term:`connection` would create a cycle, assuming that no cycle already exists in the graph represented
    by ``connections``. Used to avoid :term:`recurrent` networks when a purely :term:`feed-forward` network is desired (e.g., as determined by the
    ``feed_forward`` setting in the :ref:`configuration file <feed-forward-config-label>`.

    :param connections: The current network, as a list of (input, output) connection :term:`identifiers <key>`.
    :type connections: list(tuple(int, int))
    :param test: Possible connection to be checked for causing a cycle.
    :type test: tuple(int, int)
    :return: True if a cycle would be created; false if not.
    :rtype: bool

  .. py:function:: required_for_output(inputs, outputs, connections)

    Collect the :term:`nodes <node>` whose state is required to compute the final network output(s).

    :param inputs: the :term:`input node` :term`identifiers <key>`; **it is assumed that the input identifier set and the node identifier set are disjoint.**
    :type inputs: list(int)
    :param outputs: the :term:`output node` identifiers; by convention, the output node :term:`ids <key>` are always the same as the output index.
    :type outputs: list(int)
    :param connections: list of (input, output) connections in the network; should only include enabled ones.
    :type connections: list(tuple(int, int))
    :return: A list of layers, with each layer consisting of a set of node identifiers.
    :rtype: list(set(int))

  .. py:function:: feed_forward_layers(inputs, outputs, connections)

    Collect the layers whose members can be evaluated in parallel in a :term:`feed-forward` network.

    :param inputs: the network :term:`input node` :term:`identifiers <key>`.
    :type inputs: list(int)
    :param outputs: the :term:`output node` :term:`identifiers <key>`.
    :type outputs: list(int)
    :param connections: list of (input, output) connections in the network; should only include enabled ones.
    :type connections: list(tuple(int, int))
    :return: A list of layers, with each layer consisting of a set of :term:`identifiers <key>`; only includes nodes returned by required_for_output.
    :rtype: list(set(int))

.. py:module:: indexer
   :synopsis: Contains the Indexer class, to help with creating new identifiers/keys.

.. index:: ! key
.. index::
  see: id; key

indexer
----------

  .. py:class:: Indexer(first)

    Initializes an Indexer instance with the internal ID counter set to ``first``. This class functions to help with creating new (unique) identifiers/keys.

    :param int first: The initial identifier (:term:`key`) to be used.

    .. py:method:: get_next(result=None)

      If ``result`` is not `None`, then we return it unmodified.  Otherwise, we return the next ID and increment our internal counter.

      :param result: Returned unmodified unless `None`.
      :type result: int or None
      :return: Identifier/:term:`key` to use.
      :rtype: int

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

  .. index:: node
  .. index:: gene

  .. py:class:: IZNodeGene(BaseGene)

    Contains attributes for the iznn :term:`node` genes and determines :term:`genomic distances <genomic distance>`.

  .. index:: genome

  .. py:class:: IZGenome(DefaultGenome)

    Contains the parse_config class method for iznn genome configuration.

  .. py:class:: IZNeuron(bias, a, b, c, d, inputs)

    Sets up and simulates the iznn :term:`nodes <node>` (neurons).

    :param float bias: The bias of the neuron.
    :param float a: The time scale of the recovery variable.
    :param float b: The sensitivity of the recovery variable.
    :param float c: The after-spike reset value of the membrane potential.
    :param float d: The after-spike reset of the recovery variable.
    :param inputs: A list of (input key, weight) pairs for incoming connections.
    :type inputs: list(tuple(int, float))

    .. py:method:: advance(dt_msec)

      Advances simulation time for the neuron by the given time step in milliseconds. TODO: Currently has some numerical stability problems.

      :param float dt_msec: Time step in milliseconds.

    .. py:method:: reset()

      Resets all state variables.

  .. py:class:: IZNN(neurons, inputs, outputs)

    Sets up the network itself and simulates it using the connections and neurons.

    :param list neurons: The :py:class:`IZNeuron` instances needed.
    :param inputs: The :term:`input node` keys.
    :type inputs: list(int)
    :param outputs: The :term:`output node` keys.
    :type outputs: list(int)

    .. py:method:: set_inputs(inputs)

      Assigns input voltages.

      :param inputs: The input voltages for the :term:`input nodes <input node>`.
      :type inputs: list(float)

    .. py:method:: reset()

      Resets all neurons to their default state.

    .. py:method:: get_time_step_msec()

      Returns a suggested time step; currently hardwired to 0.05 - investigation of this (particularly effects on numerical stability issues) is planned.

      :return: Suggested time step in milliseconds.
      :rtype: float

    .. py:method:: advance(dt_msec)

      Advances simulation time for all neurons in the network by the input number of milliseconds.

      :param float dt_msec: How many milliseconds to advance the network.
      :return: The values for the :term:`output nodes <output node>`.
      :rtype: list(float)

    .. py:staticmethod:: create(genome, config)

      Receives a genome and returns its phenotype (a neural network).

      :param object genome: An IZGenome instance.
      :param object config: Configuration object.
      :return: An IZNN instance.
      :rtype: object

.. py:module:: math_util
   :synopsis: Contains some mathematical functions not found in the Python2 standard library, plus a mechanism for looking up some commonly used functions by name.

math_util
-------------

  .. py:data:: stat_functions

    Lookup table for commonly used ``{value} -> value`` functions; includes `max`, `min`, `mean`, and `median`.

  .. py:function:: mean(values)

    Returns the arithmetic mean.

  .. py:function:: median(values)

    Returns the median. (Note: For even numbers of values, does not take the mean between the two middle values.)

  .. py:function:: variance(values)

    Returns the (population) variance.

  .. py:function:: stdev(values)

    Returns the (population) standard deviation. *Note spelling.*

  .. py:function:: softmax(values)

    Compute the softmax (a differentiable/smooth approximization of the maximum function) of the given value set.
    The softmax is defined as follows: :math:`\begin{equation}v_i = \exp(v_i) / s \text{, where } s = \sum(\exp(v_0), \exp(v_1), \dotsc)\end{equation}`.

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

      :param list inputs: The values for the :term:`input nodes <input node>`.
      :return: The values for the :term:`output nodes <output node>`.
      :rtype: list

    .. py:staticmethod:: create(genome, config)

      Receives a genome and returns its phenotype (a :py:class:`FeedForwardNetwork`).

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

      :param list inputs: The values for the :term:`input nodes <input node>`.
      :return: The values for the :term:`output nodes <output node>`.
      :rtype: list

    .. py:staticmethod:: create(genome, config)

      Receives a genome and returns its phenotype (a :py:class:`RecurrentNetwork`).

.. py:module:: parallel
   :synopsis: Runs evaluation functions in parallel subprocesses in order to evaluate multiple genomes at once.

parallel
----------

  .. py:class:: ParallelEvaluator(num_workers, eval_function, timeout=None)

    Runs evaluation functions in parallel subprocesses in order to evaluate multiple genomes at once.

    :param int num_workers: How many workers to have in the `Pool <python:multiprocessing.pool.Pool>`.
    :param eval_function: eval_function should take one argument (a genome object) and return a single float (the genome's fitness) Note that this is not the same as how a fitness function is called by :py:meth:`Population.run <population.Population.run>`.
    :type eval_function: `function`
    :param timeout: How long (in seconds) each subprocess will be given before an exception is raised (unlimited if `None`).
    :type timeout: int or None

.. todo::

  Put in more about calls to rest of program?

.. py:module:: population
   :synopsis: Implements the core evolution algorithm.

population
--------------

  .. index:: reset_on_extinction

  .. py:exception:: CompleteExtinctionException

    Raised on complete extinction (all species removed due to stagnation) unless :ref:`reset_on_extinction <reset-on-extinction-label>` is set.

  .. index:: ! fitness function
  .. index:: fitness_criterion
  .. index:: fitness_threshold
  .. index:: start_generation()
  .. index:: end_generation()
  .. index:: post_evaluate()
  .. index:: complete_extinction()
  .. index:: found_solution()

  .. py:class:: Population(config, initial_state=None)

    This class implements the core evolution algorithm:
    1. Evaluate fitness of all genomes.
    2. Check to see if the termination criterion is satisfied; exit if it is.
    3. Generate the next generation from the current population.
    4. Partition the new generation into species based on genetic similarity.
    5. Go to 1.

    :param object config: The :py:class:`Config <config.Config>` configuration object.
    :param initial_state: If supplied (such as by a method of the :py:class:`Checkpointer <checkpoint.Checkpointer>` class), a tuple of (``Population``, ``Species``, generation number)
    :type initial_state: None or tuple(object, object, int)

    .. py:method:: run(fitness_function, n=None)

      Runs NEAT's genetic algorithm for at most n generations.  If n
      is ``None``, run until solution is found or extinction occurs.

      The user-provided fitness_function must take only two arguments:
      1. The population as a list of (genome id, genome) tuples.
      2. The current configuration object.

      The return value of the fitness function is ignored, but it must assign
      a Python `float` to the ``fitness`` member of each genome.

      The fitness function is free to maintain external state, perform
      evaluations in :py:mod:`parallel`, etc.

      It is assumed that the fitness function does not modify the list of genomes,
      the genomes themselves (apart from updating the fitness member),
      or the configuration object.

      :param object fitness_function: The fitness function to use, with arguments specified above.
      :param n: The maximum number of generations to run (unlimited if ``None``).
      :type n: int or None
      :return: The best genome seen.
      :rtype: object

.. py:module:: reporting
   :synopsis: Makes possible reporter classes, which are triggered on particular events and may provide information to the user, may do something else such as checkpointing, or may do both.

reporting
-----------

  .. inheritance-diagram:: reporting checkpoint.Checkpointer statistics.StatisticsReporter

  .. py:class:: ReporterSet

    Keeps track of the set of reporters and gives methods to dispatch them at appropriate points.

    .. py:method:: add(reporter)

      Adds a reporter to those to be called via :py:class:`ReporterSet` methods.

      :param object reporter: A reporter instance.

    .. py:method:: remove(reporter)

      Removes a reporter from those to be called via :py:class:`ReporterSet` methods.

      :param object reporter: A reporter instance.

    .. py:method:: start_generation(gen)

      Calls :py:meth:`start_generation <BaseReporter.start_generation>` on each reporter in the set.

      :param int gen: The generation number.

    .. py:method:: end_generation(config, population, species)

      Calls :py:meth:`end_generation <BaseReporter.end_generation>` on each reporter in the set.

      :param object config: :py:class:`Config <config.Config>` configuration object.
      :param population: Current population, as a dict of unique genome :term:`ID/key <key>` vs genome.
      :type population: dict(int, object)
      :param object species: Current species set object, such as a :py:class:`DefaultSpeciesSet <species.DefaultSpeciesSet>`.

    .. py:method:: post_evaluate(config, population, species)

      Calls :py:meth:`post_evaluate <BaseReporter.post_evaluate>` on each reporter in the set.

      :param object config: :py:class:`Config <config.Config>` configuration object.
      :param population: Current population, as a dict of unique genome :term:`ID/key <key>` vs genome.
      :type population: dict(int, object)
      :param object species: Current species set object, such as a :py:class:`DefaultSpeciesSet <species.DefaultSpeciesSet>`.
      :param object best_genome: The currently highest-fitness :term:`genome`. (Ties are resolved pseudorandomly, by `dictionary <dict>` ordering.)

    .. py:method:: post_reproduction(config, population, species)

       Not currently called. Would call :py:meth:`post_reproduction <BaseReporter.post_reproduction>` on each reporter in the set.

    .. py:method:: complete_extinction()

      Calls :py:meth:`complete_extinction <BaseReporter.complete_extinction>` on each reporter in the set.

    .. py:method:: found_solution(config, generation, best)

      Calls :py:meth:`found_solution <BaseReporter.found_solution>` on each reporter in the set.

      :param object config: :py:class:`Config <config.Config>` configuration object.
      :param int generation: Generation number.
      :param object best: The currently highest-fitness :term:`genome`. (Ties are resolved pseudorandomly by `dictionary <dict>` ordering.)

    .. py:method:: species_stagnant(sid, species)

      Calls :py:meth:`species_stagnant <BaseReporter.species_stagnant>` on each reporter in the set.

      :param int sid: The species :term:`id/key <key>`.
      :param object species: The :py:class:`Species <species.Species>` object.

    .. py:method:: info(msg)

      Calls :py:meth:`info <BaseReporter.info>` on each reporter in the set.

      :param str msg: Message to be handled.

  .. py:class:: BaseReporter

    Abstract class defining the reporter interface expected by ReporterSet. Inheriting from it will provide a set of ``dummy`` methods to be overridden as
    desired, as follows:

    .. py:method:: start_generation(generation)

      Called via :py:class:`ReporterSet` (by :py:meth:`population.Population.run`) at the start of each generation, prior to the invocation of the fitness function.

      :param int generation: The generation number.

    .. index:: key

    .. py:method:: end_generation(config, population, species)

      Called via :py:class:`ReporterSet` (by :py:meth:`population.Population.run`) at the end of each generation, after reproduction and speciation.

      :param object config: :py:class:`Config <config.Config>` configuration object.
      :param population: Current population, as a dict of unique genome :term:`ID/key <key>` vs genome.
      :type population: dict(int, object)
      :param object species: Current species set object, such as a :py:class:`DefaultSpeciesSet <species.DefaultSpeciesSet>`.

    .. py:method:: post_evaluate(config, population, species, best_genome)

      Called via :py:class:`ReporterSet` (by :py:meth:`population.Population.run`) after the fitness function is finished.

      :param object config: :py:class:`Config <config.Config>` configuration object.
      :param population: Current population, as a dict of unique genome :term:`ID/key <key>` vs genome.
      :type population: dict(int, object)
      :param object species: Current species set object, such as a :py:class:`DefaultSpeciesSet <species.DefaultSpeciesSet>`.
      :param object best_genome: The currently highest-fitness :term:`genome`. (Ties are resolved pseudorandomly, by `dictionary <dict>` ordering.)

    .. py:method:: post_reproduction(config, population, species)

      Not currently called (indirectly or directly), including by either :py:meth:`population.Population.run` or :py:class:`reproduction.DefaultReproduction`.
      Note: New members of the population likely will not have a set species.

    .. py:method:: complete_extinction()

      Called via :py:class:`ReporterSet` (by :py:meth:`population.Population.run`) if complete extinction (due to stagnation) occurs, prior to
      (depending on the :ref:`reset_on_extinction <reset-on-extinction-label>` configuration setting)
      a new population being created or a :py:exc:`population.CompleteExtinctionException` being raised.

    .. index:: ! found_solution()
    .. index:: fitness_threshold

    .. py:method:: found_solution(config, generation, best)

      Called via :py:class:`ReporterSet` (by :py:meth:`population.Population.run`) prior to exiting if the configured
      :ref:`fitness threshold <fitness-threshold-label>` is met. (Note: Not called upon reaching the generation maximum - set when
      calling :py:meth:`population.Population.run` - and exiting for this reason.)

      :param object config: :py:class:`Config <config.Config>` configuration object.
      :param int generation: Generation number.
      :param object best: The currently highest-fitness :term:`genome`. (Ties are resolved pseudorandomly by `dictionary <dict>` ordering.)

    .. py:method:: species_stagnant(sid, species)

      Called via :py:class:`ReporterSet` (by py:meth:`reproduction.DefaultReproduction.reproduce`) for each species considered stagnant by the
      stagnation class (such as :py:class:`stagnation.DefaultStagnation`).

      :param int sid: The species :term:`id/key <key>`.
      :param object species: The :py:class:`Species <species.Species>` object.

    .. py:method:: info(msg)

      Miscellaneous informational messages, from multiple parts of the library, called via :py:class:`ReporterSet`.

      :param str msg: Message to be handled.

  .. py:class:: StdOutReporter(show_species_detail)

    Uses print to output information about the run; an example reporter class.

    :param bool show_species_detail: Whether or not to show additional details about each species in the population.

.. py:module:: reproduction
   :synopsis: Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents.

reproduction
-----------------

  .. py:class:: DefaultReproduction(config, reporters, stagnation)

    Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents. Implements the default NEAT-python reproduction
    scheme: explicit fitness sharing with fixed-time species stagnation. For class requirements, see :ref:`reproduction-interface-label`.

    :param dict config: Configuration object, in this implementation a dictionary.
    :param object reporters: A :py:class:`ReporterSet <reporting.ReporterSet>` object.
    :param object stagnation: A :py:class:`DefaultStagnation <stagnation.DefaultStagnation>` object - the current code partially depends on internals of this class (a TODO is noted to correct this)

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Provides defaults for :index:`elitism`, :index:`survival_threshold`, and :index:`min_species_size` parameters and updates them from the
      :ref:`configuration file <reproduction-config-label>`.

      :param dict param_dict: Dictionary of parameters from configuration file.
      :return: Configuration object; considered opaque by rest of code, so current type returned is not required for interface.
      :rtype: dict

    .. py:classmethod:: write_config(f, param_dict)

      Required interface method. Saves ``elitism`` and ``survival_threshold`` (but not ``min_species_size``) parameters to new config file.

      :param f: `File object <file>` to write to.
      :type f: `file`
      :param dict param_dict: Dictionary of current parameters in this implementation; more generally, reproduction config object.

    .. index:: genome

    .. py:method:: create_new(genome_type, genome_config, num_genomes)

      Required interface method. Creates ``num_genomes`` new genomes of the given type using the given configuration. Also initializes ancestry
      information (as an empty tuple).

      :param genome_type: Genome class (such as :py:class:`DefaultGenome <genome.DefaultGenome>` or :py:class:`iznn.IZGenome`) of which to create instances.
      :type genome_type: `class`
      :param object genome_config: Opaque genome configuration object.
      :param int num_genomes: How many new genomes to create.
      :return: A dictionary (with the unique genome identifier as the key) of the genomes created.
      :rtype: dict(int, object)

    .. py:staticmethod:: compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size)

      Apportions desired number of members per species according to fitness (adjusted by :py:meth:`reproduce` to a 0-1 scale) from out of the
      desired population size.

      :param adjusted_fitness: Mean fitness for species members, adjusted to 0-1 scale (see below).
      :type adjusted_fitness: list(float)
      :param previous_sizes: Number of members of species in population prior to reproduction.
      :type previous_sizes: list(int)
      :param int pop_size: Desired population size, as input to :py:meth:`reproduce`.
      :param int min_species_size: Minimum number of members per species; can result in population size being above ``pop_size``.

    .. index:: ! pop_size
    .. index:: ! fitness function
    .. index:: key
    .. index:: ! elitism
    .. index:: ! survival_threshold
    .. index:: ! species_stagnant()
    .. index:: stagnation
    .. index:: ! info()

    .. py:method:: reproduce(config, species, pop_size, generation)

      Required interface method. Creates the population to be used in the next generation from the given configuration instance, SpeciesSet instance,
      desired :index:`size of the population <pop_size>`, and current generation number.  This method is called after all genomes have been evaluated and
      their ``fitness`` member assigned.  This method should use the stagnation instance given to the initializer to remove species deemed to have stagnated.
      Note: Determines relative fitnesses by transforming into (ideally) a 0-1 scale; however, if the top and bottom fitnesses are not at least 1 apart, the
      range may be less than 0-1, as a check against dividing by a too-small number. TODO: Make minimum difference configurable (defaulting to 1 to
      preserve compatibility).

      :param object config: A :py:class:`Config <config.Config>` instance.
      :param object species: A :py:class:`DefaultSpeciesSet <species.DefaultSpeciesSet>` instance. As well as depending on some of the :py:class:`DefaultStagnation <stagnation.DefaultStagnation>` internals, this method also depends on some of those of the ``DefaultSpeciesSet`` and its referenced species objects.
      :param int pop_size: Population size desired, such as set in the :ref:`configuration file <pop-size-label>`.
      :param int generation: Generation count.
      :return: New population, as a dict of unique genome :term:`ID/key <key>` vs :term:`genome`.
      :rtype: dict(int, object)

.. todo::
  Better documentation for the ``kw`` parameter in the below. Internally, these are using ``**kw`` as a **parameter** for
  keys/items/values/iterkeys/iteritems/itervalues! Is this in case someone puts in a set of key/value pairs instead of a dictionary?
  The `six documentation <https://pythonhosted.org/six/>`_ just states that this parameter is "passed to the underlying method", which is not helpful.

.. py:module:: six_util
   :synopsis: Provides Python 2/3 portability with three dictionary iterators; copied from the `six` module.

six_util
----------

This Python 2/3 portability code was copied from the `six module <https://pythonhosted.org/six/>`_ to avoid adding it as a dependency.

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
   :synopsis: Divides the population into genome-based species.

species
-----------

  .. py:class:: Species(key, generation)

    Represents a :term:`species` and contains data about it such as members, fitness, and time stagnating.
    Note: :py:class:`stagnation.DefaultStagnation` manipulates many of these.

    :param int key: :term:`Identifier/key <key>`
    :param int generation: Initial generation of appearance

  .. index:: ! genomic distance

  .. py:class:: GenomeDistanceCache(config)

    Caches (indexing by :term:`genome` :term:`key`/id) :term:`genomic distance` information to avoid repeated lookups. (The :py:meth:`distance function
    <genome.DefaultGenome.distance>` is among the most time-consuming parts of the library, although many fitness functions are likely to far outweigh
    this for moderate-size populations.)

    :param object config: A genome configuration object; later used by the genome distance function.

    .. py:method:: __call__(genome0, genome1)

      GenomeDistanceCache is called as a method with a pair of genomes to retrieve the distance.

      :param object genome0: The first genome object.
      :param object genome1: The second genome object.
      :return: The :term:`genomic distance`.
      :rtype: float

  .. py:class:: DefaultSpeciesSet(config, reporters)

    Encapsulates the default speciation scheme by configuring it and performing the speciation function (placing genomes into species by genetic similarity).
    :py:class:`reproduction.DefaultReproduction` currently depends on this having a ``species`` attribute consisting of a dictionary of species keys to species.

    :param object config: A configuration object (currently unused).
    :param object reporters: A :py:class:`ReporterSet <reporting.ReporterSet>` instance giving reporters to be notified about :term:`genomic distance` statistics.

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Currently, the only configuration parameter is the :ref:`compatibility_threshold <compatibility-threshold-label>`.

      :param param_dict: Dictionary of parameters from configuration file.
      :type param_dict: dict(str, str)
      :return: Configuration object; considered opaque by rest of code, so current type returned is not required for interface.
      :rtype: dict

    .. py:classmethod:: write_config(f, param_dict)

      Required interface method. Writes parameter(s) to new config file.

      :param f: `File object <file>` to write to.
      :type f: `file`
      :param dict param_dict: Dictionary of current parameters in this implementation; more generally, stagnation config object.

    .. index:: ! genomic distance
    .. index:: compatibility_threshold
    .. index:: info()

    .. py:method:: speciate(config, population, generation)

      Required interface method. Place genomes into species by genetic similarity (:term:`genomic distance`). (The current code has a `docstring` stating
      that there may be a problem if all old species representatives are not dropped for each generation; it is not clear how this is consistent with the code
      in :py:meth:`reproduction.DefaultReproduction.reproduce`, such as for ``elitism``.)

      :param object config: :py:class:`Config <config.Config>` object.
      :param population: Population as per the output of :py:meth:`DefaultReproduction.reproduce <reproduction.DefaultReproduction.reproduce>`.
      :type population: dict(int, object)
      :param int generation: Current generation number.

    .. py:method:: get_species_id(individual_id)

      Required interface method (used by :py:class:`reporting.StdOutReporter`). Retrieves species :term:`id/key <key>` for a given genome id/key.

      :param int individual_id: Genome id/:term:`key`.
      :return: Species id/:term:`key`.
      :rtype: int

    .. py:method:: get_species(individual_id)

      Retrieves species object for a given genome :term:`id/key <key>`. May become a required interface method, and useful for some fitness
      functions already.

      :param int individual_id: Genome id/:term:`key`.
      :return: :py:class:`Species <species.Species>` containing the genome corresponding to the id/key.
      :rtype: object

.. todo::

   ADD more methods to the below for DefaultStagnation; try to figure out which ones are required interface methods; links re config file.

.. index:: ! species_fitness_func
.. index:: fitness_criterion
.. index:: fitness_threshold

.. note::

  TODO: Currently, depending on the settings for :ref:`species_fitness_func <species-fitness-func-label>` and
  :ref:`fitness_criterion <fitness-criterion-label>`, it is possible for a species with members **above** the :ref:`fitness_threshold <fitness-threshold-label>`
  level of fitness to be considered "stagnant" (including, in particular, because they are at the limit of fitness improvement).

.. py:module:: stagnation
   :synopsis: Keeps track of whether species are making progress and removes ones that are not (for a configurable number of generations).

stagnation
--------------

  .. index:: ! max_stagnation
  .. index:: ! species_elitism

  .. py:class:: DefaultStagnation(config, reporters)

    Keeps track of whether species are making progress and helps remove ones that, for a configurable number of generations, are not.

    :param object config: Configuration object; in this implementation, a `dictionary <dict>`, but should be treated as opaque outside this class.
    :param reporters: A :py:class:`ReporterSet <reporting.ReporterSet>` with reporters that may need activating; not currently used.
    :type reporters: `class`

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Provides defaults for ``species_fitness_func``, ``max_stagnation``, and ``species_elitism`` parameters and updates them
      from the configuration file.

      :param param_dict: Dictionary of parameters from configuration file.
      :type param_dict: dict(str, str)
      :return: Configuration object; considered opaque by rest of code, so current type returned is not required for interface.
      :rtype: dict

    .. py:classmethod:: write_config(f, param_dict)

      Required interface method. Saves parameters to new config file. **Has a default of 15 for species_elitism, but will be overridden by the default of
      0 in parse_config.**

      :param f: `File object <file>` to write to.
      :type f: `file`
      :param dict param_dict: Dictionary of current parameters in this implementation; more generally, stagnation config object.

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

      Gets the per-generation average fitness. A wrapper for :py:meth:`get_fitness_stat` with the function being ``mean``.

      :return: List of mean genome fitnesses for each generation.
      :rtype: list(float)

    .. py:method:: get_fitness_stdev()

      Gets the per-generation standard deviation of the fitness. A wrapper for :py:meth:`get_fitness_stat` with the function being ``stdev``.

      :return: List of standard deviations of genome fitnesses for each generation.
      :rtype: list(float)

    .. py:method:: best_unique_genomes(n)

      Returns the ``n`` most-fit genomes, with no duplication (from the most-fit genome passing unaltered to the next generation), sorted in decreasing
      fitness order.

      :param int n: Number of most-fit genomes to return.
      :return: List of ``n`` most-fit genomes (as genome objects).
      :rtype: list(object)

    .. py:method:: best_genomes(n)

      Returns the ``n`` most-fit genomes, possibly with duplicates, sorted in decreasing fitness order.

      :param int n: Number of most-fit genomes to return.
      :return: List of ``n`` most-fit genomes (as genome objects).
      :rtype: list(object)

    .. py:method:: best_genome()

      Returns the most-fit genome ever seen. A wrapper around :py:meth:`best_genomes`.

      :return: The most-fit genome.
      :rtype: object

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
