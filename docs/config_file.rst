
Configuration file description
==============================

The configuration file is in the format described in the `Python ConfigParser documentation
<https://docs.python.org/2/library/configparser.html>`_ as "a basic configuration file parser language
which provides a structure similar to what you would find on Microsoft Windows INI files."

All settings must be explicitly enumerated in the configuration file.  This makes it less likely
that library code changes will result in your project silently using different NEAT settings.  However,
it is not necessary that they appear in any certain order.

[NEAT] section
--------------

The `NEAT` section specifies parameters particular to the generic NEAT algorithm or the experiment
itself.  This section is always required.

* *max_fitness_threshold*
    When at least one individual's measured fitness exceeds this threshold, the evolution process will terminate.
* *pop_size*
    The number of individuals in each generation.
* *reset_on_extinction*
    If this evalutes to **True**, when all species simultaneously become extinct due to stagnation, a new random
    population will be created. If **False**, a *CompleteExtinctionException* will be thrown.

[DefaultStagnation] section
---------------------------

The `DefaultStagnation` section specifies parameters for the builtin `DefaultStagnation` class.
This section is only necessary if you specify this class as the stagnation implementation when
creating the `Config` instance; otherwise you need to include whatever configuration (if any) is
required for your particular implementation.

* *species_fitness_func*
    The function used to compute species fitness.  Allowed values are max, min, mean, median.
* *max_stagnation*
    Species that have not shown improvement in more than this number of generations will be considered stagnant and removed.

[DefaultReproduction] section
-----------------------------

The `DefaultReproduction` section specifies parameters for the builtin `DefaultReproduction` class.
This section is only necessary if you specify this class as the reproduction implementation when
creating the `Config` instance; otherwise you need to include whatever configuration (if any) is
required for your particular implementation.

* *elitism*
    The number of most fit individuals in each species that will be preserved as-is from one generation to the next.
* *survival_threshold*
    The fraction for each species allowed to reproduce on each generation.

[DefaultGenome] section
-----------------------

The `DefaultGenome` section specifies parameters for the builtin `DefaultGenome` class.
This section is only necessary if you specify this class as the genome implementation when
creating the `Config` instance; otherwise you need to include whatever configuration (if any) is
required for your particular implementation.

* *activation_default*
    The default activation function assigned to new nodes.
* *activation_mutate_rate*
    The probability that mutation will change the node's activation function. Valid values are on [0.0, 1.0].
* *activation_options*
    A space-separated list of the activation functions that may be used in constructing networks.  The
    set of available functions can be found here: :ref:`activation-functions-label`

* *aggregation_default*
    The default aggregation function assigned to new nodes.
* *aggregation_mutate_rate*
    The probability that mutation will change the node's aggregation function. Valid values are on [0.0, 1.0].
* *aggregation_options*
    A space-separated list of the aggregation functions that may be used in constructing networks.  The
    set of available functions is: sum, product

* *bias_init_mean*
    The mean of the normal distribution used to select bias values for new nodes.
* *bias_init_stdev*
    The standard deviation of the normal distribution used to select bias values for new nodes.
* *bias_max_value*
    The maximum allowed bias value.  Biases above this value will be clamped to this value.
* *bias_min_value*
    The minimum allowed bias value.  Biases blow this value will be clamped to this value.
* *bias_mutate_power*
    The standard deviation of the zero-centered normal distribution from which a bias value mutation is drawn.
* *bias_mutate_rate*
    The probability that mutation will change the bias of a node by adding a random value.
* *bias_replace_rate*
    The probability that mutation will replace the bias of a node with a newly chosen random value.

* *compatibility_threshold*
    Individuals whose genomic distance is less than this threshold are considered to be in the same species.
* *compatibility_disjoint_coefficient*
    The coefficient for the disjoint gene count's contribution to the genomic distance.
* *compatibility_weight_coefficient*
    The coefficient for the average weight difference's contribution to the genomic distance.

* *conn_add_prob*
    The probability that mutation will add a connection between existing nodes. Valid values are on [0.0, 1.0].
* *conn_delete_prob*
    The probability that mutation will delete an existing connection. Valid values are on [0.0, 1.0].

* *enabled_default*
    The default enabled status of newly created connections.  Valid values are True and False.
* *enabled_mutate_rate*
    The probability that mutation will toggle the enabled status of a connection. Valid values are on [0.0, 1.0].

* *feed_forward*
    If this evaluates to **True**, generated networks will not be allowed to have recurrent connections.  Otherwise
    they may be (but are not forced to be) recurrent.
* *initial_connection*
    Specifies the initial connectivity of newly-created genomes.  There are three allowed values:

    * *unconnected* - No connection genes are initially present.
    * *fs_neat* - One connection gene from one input to all hidden and output genes. (This is the FS-NEAT scheme.)
    * *full* - Each input gene is connected to all hidden and output genes, and each hidden gene is connected to all output genes.

* *node_add_prob*
    The probability that mutation will add a new node. Valid values are on [0.0, 1.0].
* *node_delete_prob*
    The probability that mutation will delete an existing node. Valid values are on [0.0, 1.0].

* *num_hidden*
    The number of hidden nodes to add to each genome in the initial population.
* *num_inputs*
    The number of nodes through which the network receives input.
* *num_outputs*
    The number of nodes to which the network delivers output.

* *response_init_mean*
    The mean of the normal distribution used to select response values for new nodes.
* *response_init_stdev*
    The standard deviation of the normal distribution used to select response values for new nodes.
* *response_max_value*
    The maximum allowed response value. Responses above this value will be clamped to this value.
* *response_min_value*
    The minimum allowed response value. Responses blow this value will be clamped to this value.
* *response_mutate_power*
    The standard deviation of the zero-centered normal distribution from which a response value mutation is drawn.
* *response_mutate_rate*
    The probability that mutation will change the response of a node by adding a random value.
* *response_replace_rate*
    The probability that mutation will replace the response of a node with a newly chosen random value.

* *weight_init_mean*
    The mean of the normal distribution used to select weight values for new connections.
* *weight_init_stdev*
    The standard deviation of the normal distribution used to select weight values for new connections.
* *weight_max_value*
    The maximum allowed weight value. Weights above this value will be clamped to this value.
* *weight_min_value*
    The minimum allowed weight value. Weights blow this value will be clamped to this value.
* *weight_mutate_power*
    The standard deviation of the zero-centered normal distribution from which a weight value mutation is drawn.
* *weight_mutate_rate*
    The probability that mutation will change the weight of a connection by adding a random value.
* *weight_replace_rate*
    The probability that mutation will replace the weight of a connection with a newly chosen random value.
