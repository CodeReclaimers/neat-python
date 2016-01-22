
Configuration file format
=========================

The configuration file is in the format described in the `Python ConfigParser documentation
<https://docs.python.org/2/library/configparser.html>`_.  Currently, all values must be explicitly enumerated in the
configuration file.  This makes it less likely that code changes will result in your project silently using different
NEAT settings.


[phenotype] section
-------------------

* *input_nodes*
    The number of nodes through which the network receives input.
* *hidden_nodes*
    The number of hidden nodes to add to each genome in the initial population.
* *output_nodes*
    The number of nodes to which the network delivers output.
* *initial_connection*
    Specifies the initial connectivity of newly-created genomes.  There are three allowed values:

    * *unconnected* - No connection genes are initially present.
    * *fs_neat* - One connection gene from one input to all hidden and output genes. (This is the FS-NEAT scheme.)
    * *fully_connected* - Each input gene is connected to all hidden and output genes, and each hidden gene is connected to all output genes.

* *max_weight*, *min_weight*
    Connection weights (as well as node bias and response) will be limited to this range.
* *feedforward*
    If this evaluates to **True**, generated networks will not be allowed to have recurrent connections.  Otherwise
    they may be (but are not forced to be) recurrent.
* *activation_functions*
    A space-separated list of the activation functions that may be used in constructing networks.  Allowable values
    are: *abs*, *clamped*, *exp*, *gauss*, *hat*, *identity*, *inv*, *log*, *relu*, *sigmoid*, *sin*, and *tanh*. The
    implementation of these functions can be found in the `nn module
    <https://github.com/CodeReclaimers/neat-python/blob/master/neat/nn/__init__.py>`_.
* *weight_stdev*
    The standard deviation of the zero-centered normal distribution used to generate initial and replacement weights.

[genetic] section
-----------------
* *pop_size*
    The number of individuals in each generation.
* *max_fitness_threshold*
    When at least one individual's measured fitness exceeds this threshold, the evolution process will terminate.
* *prob_add_conn*
    The probability that mutation will add a connection between existing nodes. Valid values are on [0.0, 1.0].
* *prob_add_node*
    The probability that mutation will add a new hidden node into an existing connection. Valid values are on [0.0, 1.0].
* *prob_delete_conn*
    The probability that mutation will delete an existing connection. Valid values are on [0.0, 1.0].
* *prob_delete_node*
    The probability that mutation will delete an existing hidden node and any connections to it.  Valid values are on [0.0, 1.0].
* *prob_mutate_bias*
    The probability that mutation will change the bias of a node by adding a random value.
* *bias_mutation_power*
    The standard deviation of the zero-centered normal distribution from which a bias change is drawn.
* *prob_mutate_response*
    The probability that mutation will change the response of a node by adding a random value.
* *response_mutation_power*
    The standard deviation of the zero-centered normal distribution from which a response change is drawn.
* *prob_mutate_weight*
    The probability that mutation will change the weight of a connection by adding a random value.
* *prob_mutate_activation*
    The probability that mutation will change the activation function of a hidden or output node.
* *prob_replace_weight*
    The probability that mutation will replace the weight of a connection with a new random value.
* *weight_mutation_power*
    The standard deviation of the zero-centered normal distribution from which a weight change is drawn.
* *prob_toggle_link*
    The probability that the enabled status of a connection will be toggled.
* *elitism*
    The number of most fit individuals in each species that will be preserved as-is from one generation to the next.
* *reset_on_extinction*
    If this evalutes to **True**, when all species simultaneously become extinct due to stagnation, a new random
    population will be created. If **False**, a *CompleteExtinctionException* will be thrown.

[genotype compatibility] section
--------------------------------
* *compatibility_threshold*
    Individuals whose genomic distance is less than this threshold are considered to be in the same species.
* *excess_coefficient*
    The coefficient for the excess gene count's contribution to the genomic distance.
* *disjoint_coefficient*
    The coefficient for the disjoint gene count's contribution to the genomic distance.
* *weight_coefficient*
    The coefficient for the average weight difference's contribution to the genomic distance.

[species] section
-----------------
* *survival_threshold*
    The fraction for each species allowed to reproduce on each generation.
* *max_stagnation*
    Species that have not shown improvement in more than this number of generations will be considered stagnant and removed.


