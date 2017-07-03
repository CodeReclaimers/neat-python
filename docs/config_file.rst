.. _configuration-file-description-label:

Configuration file description
==============================

.. default-role:: any

The configuration file is in the format described in the Python :py:mod:`configparser` documentation
as "a basic configuration file parser language which provides a structure similar to what you would find on Microsoft Windows INI files."

Most settings must be explicitly enumerated in the configuration file.  (This makes it less likely
that library code changes will result in your project silently using different NEAT settings. There are some defaults, as noted below, and
insofar as possible new configuration parameters will default to the existing behavior.)

.. note::

  TODO: Work on ways to make the existing defaults more integrated with the configuration code. This may involve a third, optional (defaulting to ``None``)
  parameter for `ConfigParameter`, giving a default. Such a mechanism will be needed to convert the configuration in `DefaultReproduction` and
  `DefaultStagnation` to match that in other classes/modules. It would also be needed to add, for instance, an alternative uniform-distribution initialization
  for various parameters such as weight and bias, without requiring all configuration files to be changed to specify the gaussian distribution currently in
  use. Another example would be for a configuration file variable for whether more than one addition/connection :term:`mutation` can take place at a time,
  as mentioned in genome.py.

Note that the `Config` constructor also requires you to explicitly specify the types that will be used
for the NEAT simulation.  This, again, is to help avoid silent changes in behavior.

.. _configuration-file-sections-label:

The configuration file is in several sections, of which at least one is required. However, there are no requirements for ordering within these sections, or for ordering of the sections themselves.


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
  The ``found_solution`` method is **not** called if the maximum number of generations is reached without the above threshold being passed.
  TODO: Add a new configuration parameter to ignore the above, provided a maximum number of generations is passed
  to :py:meth:`population.Population.run`, and if so call ``found_solution`` upon termination by a maximum number of generations. (Passing a value
  of ``None``, which was my first thought, will not work because that will be identical to no configuration of fitness_threshold in the config file.)

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

.. note::

  TODO: `DefaultStagnation.write_config` uses a default of 15 for ``species_elitism``, but the default by `DefaultStagnation.parse_config` is 0,
  which will override.

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

* *elitism*
    The number of most-fit individuals in each species that will be preserved as-is from one generation to the next. **This defaults to 0.**

.. index:: ! survival_threshold

* *survival_threshold*
    The fraction for each species allowed to reproduce each generation. **This defaults to 0.2.**

.. note::

  TODO: There is also a :index:`min_species_size` configuration parameter, defaulting to 2, although it is not written out by
  `DefaultReproduction.write_config`.

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

* *activation_default*
    The default :term:`activation function` :term:`attribute <attributes>` :py:meth:`assigned <attributes.StringAttribute.init_value>` to new
    :term:`nodes <node>`. **If none is given, or ``random`` is specified, one of the ``activation_options`` will be chosen at random.**
* *activation_mutate_rate*
    The probability that :term:`mutation` will replace the node's activation function with a
    :py:meth:`randomly-determined <attributes.StringAttribute.mutate_value>` member of the ``activation_options``.
    Valid values are in [0.0, 1.0].
* *activation_options*
    A space-separated list of the activation functions that may be used by nodes.  **This defaults to** :ref:`sigmoid <sigmoid-label>`. The
    built-in available functions can be found in :ref:`activation-functions-label`; more can be added as described in :ref:`customization-label`.

.. index:: aggregation function
.. index:: mutation
.. index:: node
.. index:: attributes

* *aggregation_default*
    The default :term:`aggregation function` :term:`attribute <attributes>` :py:meth:`assigned <attributes.StringAttribute.init_value>` to new
    :term:`nodes <node>`. **If none is given, or ``random`` is specified, one of the ``aggregation_options`` will be chosen at random.**
* *aggregation_mutate_rate*
    The probability that :term:`mutation` will replace the node's aggregation function with a
    :py:meth:`randomly-determined <attributes.StringAttribute.mutate_value>` member of the ``aggregation_options``.
    Valid values are in [0.0, 1.0].
* *aggregation_options*
    A space-separated list of the aggregation functions that may be used by nodes.  **This defaults to ``sum``.** The
    available functions (defined in `genome.DefaultGenomeConfig`) are: ``sum``, :py:func:`product <genome.product>`, ``min``, and ``max``

.. index:: bias
.. index:: mutation
.. index:: node
.. index:: attributes

* *bias_init_mean*
    The mean of the normal/gaussian distribution used to :py:meth:`select <attributes.FloatAttribute.init_value>` :term:`bias`
    :term:`attribute <attributes>` values for new :term:`nodes <node>`.
* *bias_init_stdev*
    The standard deviation of the normal/gaussian distribution used to select bias values for new nodes.
* *bias_max_value*
    The maximum allowed bias value.  Biases above this value will be :py:meth:`clamped <attributes.FloatAttribute.clamp>` to this value.
* *bias_min_value*
    The minimum allowed bias value.  Biases below this value will be :py:meth:`clamped <attributes.FloatAttribute.clamp>` to this value.
* *bias_mutate_power*
    The standard deviation of the zero-centered normal/gaussian distribution from which a bias value :term:`mutation` is drawn.
* *bias_mutate_rate*
    The probability that :term:`mutation` will change the bias of a node by adding a random value.
* *bias_replace_rate*
    The probability that :term:`mutation` will replace the bias of a node with a newly :py:meth:`chosen <attributes.FloatAttribute.mutate_value>` 
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

.. _conn-add-prob-label:

* *conn_add_prob*
    The probability that :term:`mutation` will add a :term:`connection` between existing :term:`nodes <node>`. Valid values are in [0.0, 1.0].
* *conn_delete_prob*
    The probability that :term:`mutation` will delete an existing connection. Valid values are in [0.0, 1.0].

.. _enabled-default-label:

.. index:: enabled
.. index:: ! enabled_default
.. index:: initial_connection
.. index:: connection
.. index:: attributes

* *enabled_default*
    The default :term:`enabled` :term:`attribute <attributes>` of newly created connections.  Valid values are ``True`` and ``False``.

.. note::
  "Newly created connections" include ones in newly-created genomes, if those have initial connections
  (from the setting of the :ref:`initial_connection <initial-connection-config-label>` variable).

.. index:: mutation

* *enabled_mutate_rate*
    The probability that :term:`mutation` will :py:func:`replace <attributes.BoolAttribute.mutate_value>` (50/50 chance of ``True`` or ``False``)
    the enabled status of a connection. Valid values are in [0.0, 1.0].

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
      FS-NEAT scheme.)
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

.. index:: mutation
.. index:: node

.. _node-add-prob-label:

* *node_add_prob*
    The probability that :term:`mutation` will add a new :term:`node` (essentially replacing an existing connection,
    the :term:`enabled` status of which will be set to ``False``). Valid values are in [0.0, 1.0].
* *node_delete_prob*
    The probability that :term:`mutation` will delete an existing node (and all connections to it). Valid values are in [0.0, 1.0].

.. _num-nodes-config-label:

.. index:: hidden node

* *num_hidden*
    The number of :term:`hidden nodes <hidden node>` to add to each genome in the initial population.

.. index:: input node

* *num_inputs*
    The number of :term:`input nodes <input node>`, through which the network receives inputs.

.. index:: output node

* *num_outputs*
    The number of :term:`output nodes <output node>`, to which the network delivers outputs.

.. index:: response
.. index:: mutation
.. index:: node
.. index:: attributes

* *response_init_mean*
    The mean of the normal/gaussian distribution used to :py:meth:`select <attributes.FloatAttribute.init_value>` :term:`response` multiplier
    :term:`attribute <attributes>` values for new :term:`nodes <node>`.
* *response_init_stdev*
    The standard deviation of the normal/gaussian distribution used to select response multipliers for new nodes.
* *response_max_value*
    The maximum allowed response multiplier. Response multipliers above this value will be :py:meth:`clamped <attributes.FloatAttribute.clamp>` to this value.
* *response_min_value*
    The minimum allowed response multiplier. Response multipliers below this value will be :py:meth:`clamped <attributes.FloatAttribute.clamp>` to this value.
* *response_mutate_power*
    The standard deviation of the zero-centered normal/gaussian distribution from which a response multiplier :term:`mutation` is drawn.
* *response_mutate_rate*
    The probability that :term:`mutation` will change the response multiplier of a node by adding a random value.
* *response_replace_rate*
    The probability that :term:`mutation` will replace the response multiplier of a node with a newly :py:meth:`chosen <attributes.FloatAttribute.mutate_value>` 
    random value (as if it were a new node).

.. index:: weight
.. index:: mutation
.. index:: connection
.. index:: attributes

* *weight_init_mean*
    The mean of the normal/gaussian distribution used to :py:meth:`select <attributes.FloatAttribute.init_value>` :term:`weight`
    :term:`attribute <attributes>` values for new :term:`connections <connection>`.
* *weight_init_stdev*
    The standard deviation of the normal/gaussian distribution used to select weight values for new connections.
* *weight_max_value*
    The maximum allowed weight value. Weights above this value will be :py:meth:`clamped <attributes.FloatAttribute.clamp>` to this value.
* *weight_min_value*
    The minimum allowed weight value. Weights below this value will be :py:meth:`clamped <attributes.FloatAttribute.clamp>` to this value.
* *weight_mutate_power*
    The standard deviation of the zero-centered normal/gaussian distribution from which a weight value :term:`mutation` is drawn.
* *weight_mutate_rate*
    The probability that :term:`mutation` will change the weight of a connection by adding a random value.
* *weight_replace_rate*
    The probability that :term:`mutation` will replace the weight of a connection with a newly py:meth:`chosen <attributes.FloatAttribute.mutate_value>`
    random value (as if it were a new connection).
