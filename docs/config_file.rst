.. _configuration-file-description-label:

Configuration file description
==============================

.. default-role:: any

The configuration file is in the format described in the Python :py:mod:`configparser` documentation
as "a basic configuration file parser language which provides a structure similar to what you would find on Microsoft Windows INI files."

Most settings must be explicitly enumerated in the configuration file.  (This makes it less likely
that library code changes will result in your project silently using different NEAT settings. There are some defaults, as noted below, and
insofar as possible new configuration parameters will default to the existing behavior.)

Note that the `Config` constructor also requires you to explicitly specify the types that will be used
for the NEAT simulation.  This, again, is to help avoid silent changes in behavior.

.. _configuration-file-sections-label:

The configuration file is in several sections, of which at least one is required. However, there are no requirements for ordering within
these sections, or for ordering of the sections themselves.

.. _configuration-file-NEAT-section-label:

[NEAT] section
--------------

The ``NEAT`` section specifies parameters particular to the generic NEAT algorithm or the experiment
itself.  This section is always required, and is handled by the `Config` class itself.

.. _fitness-criterion-label:

.. index:: ! fitness_criterion

* *fitness_criterion*
    The function used to compute the termination criterion from the set of genome fitnesses.  Allowable values are: ``min``, ``max``, and ``mean``

.. _fitness-threshold-label:

.. index:: ! fitness_threshold
.. index:: found_solution()

* *fitness_threshold*
    When the fitness computed by ``fitness_criterion`` meets or exceeds this threshold, the evolution process will terminate, with a call to
    any registered reporting class' :py:meth:`found_solution <reporting.BaseReporter.found_solution>` method.

.. note::
  The ``found_solution`` method is **not** called if the maximum number of generations is reached without the above threshold being passed
  (if attention is being paid to fitness for termination in the first place - see ``no_fitness_termination`` below).

.. _no-fitness-termination-label:

.. index:: ! no_fitness_termination
.. index:: found_solution()

* *no_fitness_termination*
    If this evaluates to ``True``, then the ``fitness_criterion`` and ``fitness_threshold`` are ignored for termination; only valid if termination by a maximum
    number of generations passed to :py:meth:`population.Population.run` is enabled, and the ``found_solution`` method **is** called upon generation
    number termination. If it evaluates to ``False``, then fitness is used to determine termination. **This defaults to "False".**

    .. versionadded:: 0.92

.. _pop-size-label:

.. index:: ! pop_size

* *pop_size*
    The number of individuals in each generation.

.. _reset-on-extinction-label:

.. index:: ! reset_on_extinction

* *reset_on_extinction*
    If this evaluates to ``True``, when all species simultaneously become extinct due to stagnation, a new random
    population will be created. If ``False``, a `CompleteExtinctionException` will be thrown.

.. index:: stagnation
.. index:: DefaultStagnation

[DefaultStagnation] section
---------------------------

The ``DefaultStagnation`` section specifies parameters for the builtin `DefaultStagnation` class.
This section is only necessary if you specify this class as the stagnation implementation when
creating the `Config` instance; otherwise you need to include whatever configuration (if any) is
required for your particular implementation.

.. _species-fitness-func-label:

.. index:: ! species_fitness_func

* *species_fitness_func*
    The function used to compute species fitness.  **This defaults to ``mean``.** Allowed values are: ``max``, ``min``, ``mean``, and
    :py:func:`median <math_util.median>`

.. note::

  This is **not** used for calculating species fitness for apportioning reproduction (which always uses ``mean``).

.. _max-stagnation-label:

.. index:: ! max_stagnation

* *max_stagnation*
    Species that have not shown improvement in more than this number of generations will be considered stagnant and removed. **This defaults to 15.**

.. _species-elitism-label:

.. index:: ! species_elitism

* *species_elitism*
    The number of species that will be protected from stagnation; mainly intended to prevent
    total extinctions caused by all species becoming stagnant before new species arise.  For example,
    a ``species_elitism`` setting of 3 will prevent the 3 species with the highest species fitness from
    being removed for stagnation regardless of the amount of time they have not shown improvement. **This defaults to 0.**

.. index:: reproduction
.. index:: DefaultReproduction

.. _reproduction-config-label:

[DefaultReproduction] section
-----------------------------

The ``DefaultReproduction`` section specifies parameters for the builtin `DefaultReproduction` class.
This section is only necessary if you specify this class as the reproduction implementation when
creating the `Config` instance; otherwise you need to include whatever configuration (if any) is
required for your particular implementation.

.. index:: ! elitism

.. _elitism-label:

* *elitism*
    The number of most-fit individuals in each species that will be preserved as-is from one generation to the next. **This defaults to 0.**

.. index:: ! survival_threshold

* *survival_threshold*
    The fraction for each species allowed to reproduce each generation. **This defaults to 0.2.**

.. index:: ! min_species_size

.. _min-species-size-label:

* *min_species_size*
    The minimum number of genomes per species after reproduction. **This defaults to 2.**

.. index:: genome
.. index:: DefaultGenome

[DefaultGenome] section
-----------------------

The ``DefaultGenome`` section specifies parameters for the builtin `DefaultGenome` class.
This section is only necessary if you specify this class as the genome implementation when
creating the `Config` instance; otherwise you need to include whatever configuration (if any) is
required for your particular implementation.

.. index:: activation function
.. index:: mutation
.. index:: node
.. index:: attributes

.. _activation-function-config-label:

.. index:: X_default

* *activation_default*
    The default :term:`activation function` :term:`attribute <attributes>` :py:meth:`assigned <attributes.StringAttribute.init_value>` to new
    :term:`nodes <node>`. **If none is given, or "random" is specified, one of the activation_options will be chosen at random.**

.. index:: mutate_rate

* *activation_mutate_rate*
    The probability that :term:`mutation` will replace the node's activation function with a
    :py:meth:`randomly-determined <attributes.StringAttribute.mutate_value>` member of the ``activation_options``.
    Valid values are in [0.0, 1.0].

.. index:: X_options

* *activation_options*
    A space-separated list of the activation functions that may be used by nodes.  **This defaults to** :ref:`sigmoid <sigmoid-label>`. The
    built-in available functions can be found in :ref:`activation-functions-label`; more can be added as described in :ref:`customization-label`.

.. index:: aggregation function
.. index:: mutation
.. index:: node
.. index:: attributes

.. _aggregation-function-config-label:

.. index:: X_default

* *aggregation_default*
    The default :term:`aggregation function` :term:`attribute <attributes>` :py:meth:`assigned <attributes.StringAttribute.init_value>` to new
    :term:`nodes <node>`. **If none is given, or "random" is specified, one of the aggregation_options will be chosen at random.**

.. index:: mutate_rate

* *aggregation_mutate_rate*
    The probability that :term:`mutation` will replace the node's aggregation function with a
    :py:meth:`randomly-determined <attributes.StringAttribute.mutate_value>` member of the ``aggregation_options``.
    Valid values are in [0.0, 1.0].

.. index:: X_options

* *aggregation_options*
    A space-separated list of the aggregation functions that may be used by nodes.  **This defaults to "sum".** The
    available functions (defined in `aggregations`) are: ``sum``, :py:func:`product <aggregations.product_aggregation>`, ``min``, ``max``, ``mean``, ``median``,
    and :py:func:`maxabs <aggregations.maxabs_aggregation>` (which returns the input value with the greatest absolute value; the returned
    value may be positive or negative). New aggregation functions can be defined similarly to :ref:`new activation functions <customization-label>`.
    (Note that the function needs to take a `list` or other `iterable`; the `reduce <functools.reduce>` function, as in `aggregations`, may be of use in this.)

    .. versionchanged:: 0.92
      Moved out of :py:mod:`genome` into :py:mod:`aggregations`; maxabs, mean, and median added; method for defining new aggregation functions added.

.. index:: bias
.. index:: mutation
.. index:: node
.. index:: attributes

.. index:: init_mean

* *bias_init_mean*
    The mean of the normal/gaussian distribution, if it is used to :py:meth:`select <attributes.FloatAttribute.init_value>` :term:`bias`
    :term:`attribute <attributes>` values for new :term:`nodes <node>`.

.. index:: init_stdev

* *bias_init_stdev*
    The standard deviation of the normal/gaussian distribution, if it is used to select bias values for new nodes.

.. index:: init_type

* *bias_init_type*
    If set to ``gaussian`` or ``normal``, then the initialization is to a normal/gaussian distribution. If set to ``uniform``, a uniform distribution
    from :math:`\max(bias\_min\_value, (bias\_init\_mean-(bias\_init\_stdev*2)))` to
    :math:`\min(bias\_max\_value, (bias\_init\_mean+(bias\_init\_stdev*2)))`. (Note that the standard deviation of a uniform distribution is not
    range/0.25, as implied by this, but the range divided by a bit over 0.288 (the square root of 12); however, this approximation makes setting
    the range much easier.) **This defaults to "gaussian".**

    .. versionadded:: 0.92

.. index:: max_value
.. index:: min_value

* *bias_max_value*
    The maximum allowed bias value.  Biases above this value will be :py:meth:`clamped <attributes.FloatAttribute.clamp>` to this value.
* *bias_min_value*
    The minimum allowed bias value.  Biases below this value will be :py:meth:`clamped <attributes.FloatAttribute.clamp>` to this value.

.. index:: mutate_power

* *bias_mutate_power*
    The standard deviation of the zero-centered normal/gaussian distribution from which a bias value :term:`mutation` is drawn.

.. index:: mutate_rate

* *bias_mutate_rate*
    The probability that :term:`mutation` will change the bias of a node by adding a random value.

.. index:: replace_rate

* *bias_replace_rate*
    The probability that :term:`mutation` will replace the bias of a node with a newly :py:meth:`chosen <attributes.FloatAttribute.init_value>`
    random value (as if it were a new node).

.. _compatibility-threshold-label:

.. index:: genomic distance
.. index:: ! compatibility_threshold
.. index:: species

* *compatibility_threshold*
    Individuals whose :term:`genomic distance` is less than this threshold are considered to be in the same :term:`species`.

.. _compatibility-disjoint-coefficient-label:

.. index:: ! compatibility_disjoint_coefficient
.. index:: disjoint

* *compatibility_disjoint_coefficient*
    The coefficient for the :term:`disjoint` and :term:`excess` :term:`gene` counts' contribution to the :term:`genomic distance`.

.. _compatibility-weight-coefficient-label:

.. index:: ! compatibility_weight_coefficient
.. index:: attributes
.. index:: homologous

* *compatibility_weight_coefficient*
    The coefficient for each :term:`weight`, :term:`bias`, or :term:`response` multiplier difference's contribution to the :term:`genomic distance`
    (for :term:`homologous` :term:`nodes <node>` or :term:`connections <connection>`). This is also used as the value to add for differences
    in :term:`activation functions <activation function>`, :term:`aggregation functions <aggregation function>`, or :term:`enabled`/disabled status.

.. note::
  It is currently possible for two :term:`homologous` nodes or connections to have a higher contribution to the :term:`genomic distance` than a
  disjoint or excess :term:`node` or :term:`connection`, depending on their :term:`attributes` and the settings of the above parameters.

.. index:: mutation
.. index:: connection
.. index:: conn_add_prob

.. _conn-add-prob-label:

* *conn_add_prob*
    The probability that :term:`mutation` will add a :term:`connection` between existing :term:`nodes <node>`. Valid values are in [0.0, 1.0].

.. index:: conn_delete_prob

* *conn_delete_prob*
    The probability that :term:`mutation` will delete an existing connection. Valid values are in [0.0, 1.0].

.. _enabled-default-label:

.. index:: enabled
.. index:: ! enabled_default
.. index:: initial_connection
.. index:: connection
.. index:: attributes

.. index:: X_default

* *enabled_default*
    The default :term:`enabled` :term:`attribute <attributes>` of newly created connections.  Valid values are ``True`` and ``False``.

.. note::
  "Newly created connections" include ones in newly-created genomes, if those have initial connections
  (from the setting of the :ref:`initial_connection <initial-connection-config-label>` variable).

.. index:: mutation
.. index:: mutate_rate

* *enabled_mutate_rate*
    The probability that :term:`mutation` will :py:func:`replace <attributes.BoolAttribute.mutate_value>` (50/50 chance of ``True`` or ``False``)
    the enabled status of a connection. Valid values are in [0.0, 1.0].

.. index:: rate_to_false_add
.. index:: rate_to_true_add

* *enabled_rate_to_false_add*
    Adds to the ``enabled_mutate_rate`` if the connection is currently :term:`enabled`.
* *enabled_rate_to_true_add*
    Adds to the ``enabled_mutate_rate`` if the connection is currently not enabled.

    .. versionadded:: 0.92
      ``enabled_rate_to_false_add`` and ``enabled_rate_to_true_add``

.. _feed-forward-config-label:

.. index:: ! feed_forward
.. index:: feedforward

* *feed_forward*
    If this evaluates to ``True``, generated networks will not be allowed to have :term:`recurrent` :term:`connections <connection>`
    (they will be :term:`feedforward`). Otherwise they may be (but are not forced to be) recurrent.

.. _initial-connection-config-label:

.. index:: ! initial_connection
.. index:: enabled_default
.. index:: connection

* *initial_connection*
    Specifies the initial connectivity of newly-created genomes.  (Note the effects on settings other than ``unconnected`` of the
    :ref:`enabled_default <enabled-default-label>` parameter.) There are seven allowed values:

    * ``unconnected`` - No :term:`connections <connection>` are initially present. **This is the default.**
    * ``fs_neat_nohidden`` - One randomly-chosen :term:`input node` has one connection to each :term:`output node`. (This is one version of the
      FS-NEAT scheme; "FS" stands for "Feature Selection".)
    * ``fs_neat_hidden`` - One randomly-chosen :term:`input node` has one connection to each :term:`hidden <hidden node>` and
      :term:`output node`. (This is another version of the FS-NEAT scheme. If there are no hidden nodes, it is the same as ``fs_neat_nohidden``.)
    * ``full_nodirect`` - Each :term:`input node` is connected to all :term:`hidden <hidden node>` nodes, if there are any, and each hidden node is
      connected to all :term:`output nodes <output node>`; otherwise, each input node is connected to all :term:`output nodes <output node>`.
      Genomes with :ref:`feed_forward <feed-forward-config-label>` set to ``False`` will also have :term:`recurrent` (loopback, in this case)
      connections from each hidden or output node to itself.
    * ``full_direct`` - Each :term:`input node` is connected to all :term:`hidden <hidden node>` and :term:`output nodes <output node>`,
      and each hidden node is connected to all output nodes. Genomes with :ref:`feed_forward <feed-forward-config-label>` set to ``False`` will also
      have :term:`recurrent` (loopback, in this case) connections from each hidden or output node to itself.
    * ``partial_nodirect #`` - As for ``full_nodirect``, but each connection has a probability of being present determined by the number
      (valid values are in [0.0, 1.0]).
    * ``partial_direct #`` - as for ``full_direct``, but each connection has a probability of being present determined by the number
      (valid values are in [0.0, 1.0]).

.. versionchanged:: 0.92
  fs_neat split into fs_neat_nohidden and fs_neat_hidden; full, partial split into full_nodirect, full_direct, partial_nodirect, partial_direct

.. index:: mutation
.. index:: node
.. index:: node_add_prob

.. _node-add-prob-label:

* *node_add_prob*
    The probability that :term:`mutation` will add a new :term:`node` (essentially replacing an existing connection,
    the :term:`enabled` status of which will be set to ``False``). Valid values are in [0.0, 1.0].

.. index:: node_delete_prob

* *node_delete_prob*
    The probability that :term:`mutation` will delete an existing node (and all connections to it). Valid values are in [0.0, 1.0].

.. _num-nodes-config-label:

.. index:: hidden node
.. index:: ! num_hidden

* *num_hidden*
    The number of :term:`hidden nodes <hidden node>` to add to each genome in the initial population.

.. index:: input node
.. index:: ! num_inputs

* *num_inputs*
    The number of :term:`input nodes <input node>`, through which the network receives inputs.

.. index:: output node
.. index:: ! num_outputs

* *num_outputs*
    The number of :term:`output nodes <output node>`, to which the network delivers outputs.

.. index:: response
.. index:: mutation
.. index:: node
.. index:: attributes

.. index:: init_mean

* *response_init_mean*
    The mean of the normal/gaussian distribution, if it is used to :py:meth:`select <attributes.FloatAttribute.init_value>` :term:`response` multiplier
    :term:`attribute <attributes>` values for new :term:`nodes <node>`.

.. index:: init_stdev

* *response_init_stdev*
    The standard deviation of the normal/gaussian distribution, if it is used to select response multipliers for new nodes.

.. index:: init_type

* *response_init_type*
    If set to ``gaussian`` or ``normal``, then the initialization is to a normal/gaussian distribution. If set to ``uniform``, a uniform distribution
    from :math:`\max(response\_min\_value, (response\_init\_mean-(response\_init\_stdev*2)))` to
    :math:`\min(response\_max\_value, (response\_init\_mean+(response\_init\_stdev*2)))`. (Note that the standard deviation of a uniform distribution is not
    range/0.25, as implied by this, but the range divided by a bit over 0.288 (the square root of 12); however, this approximation makes setting
    the range much easier.) **This defaults to "gaussian".**

    .. versionadded:: 0.92

.. index:: max_value
.. index:: min_value

* *response_max_value*
    The maximum allowed response multiplier. Response multipliers above this value will be :py:meth:`clamped <attributes.FloatAttribute.clamp>` to this
    value.
* *response_min_value*
    The minimum allowed response multiplier. Response multipliers below this value will be :py:meth:`clamped <attributes.FloatAttribute.clamp>` to this value.

.. index:: mutate_power

* *response_mutate_power*
    The standard deviation of the zero-centered normal/gaussian distribution from which a response multiplier :term:`mutation` is drawn.

.. index:: mutate_rate

* *response_mutate_rate*
    The probability that :term:`mutation` will change the response multiplier of a node by adding a random value.

.. index:: replace_rate

* *response_replace_rate*
    The probability that :term:`mutation` will replace the response multiplier of a node with a newly :py:meth:`chosen <attributes.FloatAttribute.init_value>` 
    random value (as if it were a new node).

.. index:: ! single_structural_mutation
.. index:: ! structural_mutation_surer
.. index:: mutation
.. index:: node
.. index:: connection

.. _structural-mutation-surer-label:

* *single_structural_mutation*
    If this evaluates to ``True``, only one structural mutation (the addition or removal of a :term:`node` or :term:`connection`) will be allowed per genome
    per generation. (If the probabilities for :ref:`conn_add_prob <conn-add-prob-label>`, conn_delete_prob, :ref:`node_add_prob <node-add-prob-label>`,
    and node_delete_prob add up to over 1, the chances of each are proportional to the appropriate configuration value.) **This defaults to "False".**

    .. versionadded:: 0.92

* *structural_mutation_surer*
    If this evaluates to ``True``, then an attempt to add a :term:`node` to a genome lacking :term:`connections <connection>` will result in adding
    a connection instead; furthermore, if an attempt to add a connection tries to add a connection that already exists, that connection will be
    :term:`enabled`. If this is set to ``default``, then it acts as if it had the same value as ``single_structural_mutation`` (above). **This defaults to "default".**

    .. versionadded:: 0.92

.. index:: weight
.. index:: mutation
.. index:: connection
.. index:: attributes

.. index:: init_mean

* *weight_init_mean*
    The mean of the normal/gaussian distribution used to :py:meth:`select <attributes.FloatAttribute.init_value>` :term:`weight`
    :term:`attribute <attributes>` values for new :term:`connections <connection>`.

.. index:: init_stdev

* *weight_init_stdev*
    The standard deviation of the normal/gaussian distribution used to select weight values for new connections.

.. index:: ! init_type

* *weight_init_type*
    If set to ``gaussian`` or ``normal``, then the initialization is to a normal/gaussian distribution. If set to ``uniform``, a uniform distribution
    from :math:`\max(weight\_min\_value, (weight\_init\_mean-(weight\_init\_stdev*2)))` to
    :math:`\min(weight\_max\_value, (weight\_init\_mean+(weight\_init\_stdev*2)))`. (Note that the standard deviation of a uniform distribution is not
    range/0.25, as implied by this, but the range divided by a bit over 0.288 (the square root of 12); however, this approximation makes setting
    the range much easier.) **This defaults to "gaussian".**

    .. versionadded:: 0.92

.. index:: max_value
.. index:: min_value

* *weight_max_value*
    The maximum allowed weight value. Weights above this value will be :py:meth:`clamped <attributes.FloatAttribute.clamp>` to this value.
* *weight_min_value*
    The minimum allowed weight value. Weights below this value will be :py:meth:`clamped <attributes.FloatAttribute.clamp>` to this value.

.. index:: mutate_power

* *weight_mutate_power*
    The standard deviation of the zero-centered normal/gaussian distribution from which a weight value :term:`mutation` is drawn.

.. index:: mutate_rate

* *weight_mutate_rate*
    The probability that :term:`mutation` will change the weight of a connection by adding a random value.

.. index:: replace_rate

* *weight_replace_rate*
    The probability that :term:`mutation` will replace the weight of a connection with a newly :py:meth:`chosen <attributes.FloatAttribute.init_value>`
    random value (as if it were a new connection).

:ref:`Table of Contents <toc-label>`
