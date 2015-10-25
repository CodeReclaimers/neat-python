
Configuration file format
=========================

The configuration file is in the format described in the `Python ConfigParser documentation
<https://docs.python.org/2/library/configparser.html>`_.  Currently, all values must be explicitly enumerated in the
configuration file.


[phenotype] section
-------------------

* *input_nodes*
    The number of nodes through which the network receives input.

* *output_nodes*
    The number of nodes to which the network delivers output.

* *fully_connected*
    If this evaluates to **True**, then all individuals in the initial population will be created with each output will
    having randomly-configured connections to all inputs. Otherwise, the initial population's members will have only
    *only one* randomly-configured connection from a random input to each input.

    In both cases, the initial networks contain no hidden nodes.

* *max_weight*, *min_weight*
    Connection weights will be limited to this range.

* *feedforward*
    If this evaluates to **True**, generated networks will not be allowed to have recurrent connections.  Otherwise
    they may be (but are not forced to be) recurrent.

* *nn_activation*
    Type of activation function to be used to build networks.

* *hidden_nodes*
* *weight_stdev*
    The standard deviation of the zero-centered normal distribution used to generate initial and replacement weights.

[genetic] section
-----------------
* *pop_size*
* *max_fitness_threshold*
    When at least one individual's measured fitness exceeds this threshold, the evolution process will terminate.
* *prob_addconn*
    The probability that mutation will add a connection between existing nodes. Valid values are on [0.0, 1.0].
* *prob_addnode*
    The probability that mutation will add a new hidden node into an existing connection. Valid values are on [0.0, 1.0].
* *prob_deleteconn*
    The probability that mutation will delete an existing connection. Valid values are on [0.0, 1.0].
* *prob_deletenode*
    The probability that mutation will delete an existing hidden node and any connections to it.  Valid values are on [0.0, 1.0].
* *prob_mutatebias*
    The probability that mutation will change the bias of a node by adding a random value.
* *bias_mutation_power*
    The standard deviation of the zero-centered normal distribution from which a bias change is drawn.
* *prob_mutate_response*
    The probability that mutation will change the response of a node by adding a random value.
* *response_mutation_power*
    The standard deviation of the zero-centered normal distribution from which a response change is drawn.
* *prob_mutate_weight*
    The probability that mutation will change the weight of a connection by adding a random value.
* *weight_mutation_power*
    The standard deviation of the zero-centered normal distribution from which a weight change is drawn.
* *prob_togglelink*
    The probability that the enabled status of a connection will be toggled.
* *elitism*
    The number of individuals in each species that will be preserved from one generation to the next.

[genotype compatibility] section
--------------------------------
* *compatibility_threshold*
* *compatibility_change*
* *excess_coefficient*
* *disjoint_coefficient*
* *weight_coefficient*

[species] section
-----------------
* *species_size*
* *survival_threshold*
* *old_threshold*
* *youth_threshold*
* *old_penalty*
* *youth_boost*
* *max_stagnation*


