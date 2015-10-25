
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
    If this evaluates to **True**, then all individuals in the initial population will be created with
    no hidden nodes, and each output will have  randomly-configured connection to each input.
    Otherwise, the initial population's members will have no hidden nodes, and each output will have
    *only one* randomly-configured connection to a random input.

* *max_weight*, *min_weight*
    Connection weights will be limited to this range.

* *feedforward*

* *nn_activation*
    Type of activation function to be used to build networks.

* *hidden_nodes*
* *weight_stdev*
    The standard deviation of the zero-centered normal distribution used to generate initial and replacement weights.

[genetic] section
-----------------
* *pop_size*
* *max_fitness_threshold*
* *prob_addconn*
* *prob_addnode*
* *prob_deleteconn*
* *prob_deletenode*
* *prob_mutatebias*
* *bias_mutation_power*
* *prob_mutate_response*
* *response_mutation_power*
* *prob_mutate_weight*
* *weight_mutation_power*
* *prob_togglelink*
* *elitism*

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


