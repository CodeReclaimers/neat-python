
Configuration file description
==============================

The configuration file is in the format described in the `Python ConfigParser documentation
<https://docs.python.org/2/library/configparser.html>`_ as "a basic configuration file parser language
which provides a structure similar to what you would find on Microsoft Windows INI files."

Most settings must be explicitly enumerated in the configuration file.  (This makes it less likely
that library code changes will result in your project silently using different NEAT settings. There are some defaults, as noted below, and
insofar as possible new configuration parameters will default to the existing behavior.)  However,
it is not necessary that they appear in any certain order.

.. I have been considering ways to make the existing defaults more integrated with the configuration code. - Allen (drallensmith)

Note that the `Config` constructor also requires you to explicitly specify the types that will be used
for the NEAT simulation.  This, again, is to help avoid silent changes in behavior.

[NEAT] section
--------------

The ``NEAT`` section specifies parameters particular to the generic NEAT algorithm or the experiment
itself.  This section is always required.

* *fitness_criterion*
    The function used to compute the termination criterion from the set of genome fitnesses.  Allowable values are: ``min``, ``max``, ``mean``
* *fitness_threshold*
    When the fitness computed by ``fitness_criterion`` meets or exceeds this threshold, the evolution process will terminate.
* *pop_size*
    The number of individuals in each generation.

.. _reset-on-extinction-label:

* *reset_on_extinction*
    If this evaluates to `True`, when all species simultaneously become extinct due to stagnation, a new random
    population will be created. If `False`, a `CompleteExtinctionException` will be thrown.


[DefaultStagnation] section
---------------------------

The ``DefaultStagnation`` section specifies parameters for the builtin `DefaultStagnation` class.
This section is only necessary if you specify this class as the stagnation implementation when
creating the `Config` instance; otherwise you need to include whatever configuration (if any) is
required for your particular implementation.

* *species_fitness_func*
    The function used to compute species fitness.  **This defaults to ``mean``.** Allowed values are: ``max``, ``min``, ``mean``, ``median``
* *max_stagnation*
    Species that have not shown improvement in more than this number of generations will be considered stagnant and removed. **This defaults to 15.**
* *species_elitism*
    The number of species that will be protected from stagnation; mainly intended to prevent
    total extinctions caused by all species becoming stagnant before new species arise.  For example,
    a ``species_elitism`` setting of 3 will prevent the 3 species with the highest species fitness from
    being removed for stagnation regardless of the amount of time they have not shown improvement. **This defaults to 0.**

.. write_config in stagnation.py uses a default of 15 for species_elitism, but the default by parse_config is 0.

[DefaultReproduction] section
-----------------------------

The ``DefaultReproduction`` section specifies parameters for the builtin `DefaultReproduction` class.
This section is only necessary if you specify this class as the reproduction implementation when
creating the `Config` instance; otherwise you need to include whatever configuration (if any) is
required for your particular implementation.

* *elitism*
    The number of most-fit individuals in each species that will be preserved as-is from one generation to the next. **This defaults to 0.**
* *survival_threshold*
    The fraction for each species allowed to reproduce each generation. **This defaults to 0.2.**

.. There is also a "min_species_size" configuration parameter, defaulting to 2, although it is not written out by write_config in reproduction.py.

[DefaultGenome] section
-----------------------

The ``DefaultGenome`` section specifies parameters for the builtin `DefaultGenome` class.
This section is only necessary if you specify this class as the genome implementation when
creating the `Config` instance; otherwise you need to include whatever configuration (if any) is
required for your particular implementation.

.. index:: activation function

* *activation_default*
    The default :term:`activation function` assigned to new :term:`nodes <node>`. **If none is given, or ``random`` is specified, one of the ``activation_options``
    will be chosen at random.**
* *activation_mutate_rate*
    The probability that mutation will replace the node's activation function with a randomly-determined member of the ``activation_options``.
    Valid values are in [0.0, 1.0].
* *activation_options*
    A space-separated list of the activation functions that may be used by nodes.  **This defaults to ``sigmoid``.** The
    available functions can be found here: :ref:`activation-functions-label`

.. index:: aggregation function

* *aggregation_default*
    The default :term:`aggregation function` assigned to new nodes. **If none is given, or ``random`` is specified, one of the ``aggregation_options``
    will be chosen at random.**
* *aggregation_mutate_rate*
    The probability that mutation will replace the node's aggregation function with a randomly-determined member of the ``aggregation_options``.
    Valid values are in [0.0, 1.0].
* *aggregation_options*
    A space-separated list of the aggregation functions that may be used by nodes.  **This defaults to ``sum``.** The
    available functions are: ``sum``, ``product``, ``min``, ``max``

* *bias_init_mean*
    The mean of the normal distribution used to select :term:`bias` values for new nodes.
* *bias_init_stdev*
    The standard deviation of the normal distribution used to select bias values for new nodes.
* *bias_max_value*
    The maximum allowed bias value.  Biases above this value will be clamped to this value.
* *bias_min_value*
    The minimum allowed bias value.  Biases below this value will be clamped to this value.
* *bias_mutate_power*
    The standard deviation of the zero-centered normal distribution from which a bias value mutation is drawn.
* *bias_mutate_rate*
    The probability that mutation will change the bias of a node by adding a random value.
* *bias_replace_rate*
    The probability that mutation will replace the bias of a node with a newly chosen random value (as if it were a new node).

* *compatibility_threshold*
    Individuals whose :term:`genomic distance` is less than this threshold are considered to be in the same :term:`species`.
* *compatibility_disjoint_coefficient*
    The coefficient for the disjoint+excess :term:`gene` counts' contribution to the genomic distance.
* *compatibility_weight_coefficient*
    The coefficient for each :term:`weight`, :term:`bias`, or :term:`response` multiplier difference's contribution to the genomic distance.

* *conn_add_prob*
    The probability that mutation will add a :term:`connection` between existing :term:`nodes <node>`. Valid values are in [0.0, 1.0].
* *conn_delete_prob*
    The probability that mutation will delete an existing connection. Valid values are in [0.0, 1.0].

* *enabled_default*
    The default :term:`enabled` status of newly created connections.  Valid values are `True` and `False`.

.. note::
   "Newly created connections" include ones in newly-created genomes, if those have initial connections.

* *enabled_mutate_rate*
    The probability that mutation will replace (50/50 chance of `True` or `False`) the enabled status of a connection.
    Valid values are in [0.0, 1.0].

* *feed_forward*
    If this evaluates to `True`, generated networks will not be allowed to have :term:`recurrent` connections (they will be :term:`feedforward`).
    Otherwise they may be (but are not forced to be) recurrent.

* *initial_connection*
    Specifies the initial connectivity of newly-created genomes.  There are four allowed values:

    * ``unconnected`` - No :term:`connections <connection>` are initially present. **This is the default.**
    * ``fs_neat`` - One randomly-chosen :term:`input node` has one connection to each :term:`hidden <hidden node>` and
      :term:`output node`. (This is the FS-NEAT scheme.)
    * ``full`` - Each :term:`input node` is connected to all :term:`hidden <hidden node>` and :term:`output nodes <output node>`,
      and each hidden node is connected to all output nodes. (Note: This does not include :term:`recurrent` connections.)
    * ``partial #`` - As for ``full``, but each connection has a probability of being present determined by the number (valid values are in [0.0, 1.0]).

* *node_add_prob*
    The probability that mutation will add a new node (essentially replacing an existing connection,
    the :term:`enabled` status of which will be set to ``False``). Valid values are in [0.0, 1.0].
* *node_delete_prob*
    The probability that mutation will delete an existing node (and all connections to it). Valid values are in [0.0, 1.0].

* *num_hidden*
    The number of :term:`hidden nodes <hidden node>` to add to each genome in the initial population.
* *num_inputs*
    The number of :term:`input nodes <input node>`, through which the network receives inputs.
* *num_outputs*
    The number of :term:`output nodes <output node>`, to which the network delivers outputs.

* *response_init_mean*
    The mean of the normal distribution used to select :term:`response` multipliers for new nodes.
* *response_init_stdev*
    The standard deviation of the normal distribution used to select response multipliers for new nodes.
* *response_max_value*
    The maximum allowed response multiplier. Response multipliers above this value will be clamped to this value.
* *response_min_value*
    The minimum allowed response multiplier. Response multipliers below this value will be clamped to this value.
* *response_mutate_power*
    The standard deviation of the zero-centered normal distribution from which a response multiplier mutation is drawn.
* *response_mutate_rate*
    The probability that mutation will change the response multiplier of a node by adding a random value.
* *response_replace_rate*
    The probability that mutation will replace the response multiplier of a node with a newly chosen random value (as if it were a new node).

* *weight_init_mean*
    The mean of the normal distribution used to select :term:`weight` values for new connections.
* *weight_init_stdev*
    The standard deviation of the normal distribution used to select weight values for new connections.
* *weight_max_value*
    The maximum allowed weight value. Weights above this value will be clamped to this value.
* *weight_min_value*
    The minimum allowed weight value. Weights below this value will be clamped to this value.
* *weight_mutate_power*
    The standard deviation of the zero-centered normal distribution from which a weight value mutation is drawn.
* *weight_mutate_rate*
    The probability that mutation will change the weight of a connection by adding a random value.
* *weight_replace_rate*
    The probability that mutation will replace the weight of a connection with a newly chosen random value (as if it were a new connection).
