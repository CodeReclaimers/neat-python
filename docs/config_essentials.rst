Configuration Essentials
========================

NEAT requires a configuration file to control evolution, but don't be overwhelmed! While the complete configuration has 40+ parameters, you only need to understand about 10-15 to get started effectively.

This guide covers the essentials. For complete documentation, see :doc:`config_file`.

The Big Picture
---------------

The configuration file has 4 main sections:

1. **[NEAT]** - Overall evolution settings (population size, when to stop)
2. **[DefaultGenome]** - Network structure and mutation rules
3. **[DefaultSpeciesSet]** - How to group similar genomes
4. **[DefaultStagnation]** & **[DefaultReproduction]** - Selection and reproduction

Think of it this way: **[NEAT]** controls the experiment, **[DefaultGenome]** controls the networks, and the others control the evolutionary process.

Essential Parameters
--------------------

Start Here: The 10 You'll Always Modify
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

   [NEAT]
   pop_size = 150                    # How many genomes per generation
   fitness_threshold = 3.9           # Stop when this fitness is reached
   fitness_criterion = max           # Use the maximum fitness
   
   [DefaultGenome]
   num_inputs = 2                    # Number of input neurons
   num_outputs = 1                   # Number of output neurons
   activation_default = sigmoid      # Activation function for neurons
   
   conn_add_prob = 0.5               # Probability of adding a connection
   node_add_prob = 0.2               # Probability of adding a node
   
   [DefaultSpeciesSet]
   compatibility_threshold = 3.0     # How similar genomes must be to group together

**When to modify these:**

- **Always** set ``num_inputs`` and ``num_outputs`` to match your problem
- **Always** set ``fitness_threshold`` based on your fitness function
- **Often** adjust ``pop_size`` (bigger = slower but more thorough)
- **Sometimes** adjust mutation probabilities to control complexity growth
- **Rarely** change ``activation_default`` (sigmoid works for most problems)

Complete Minimal Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a minimal working config file. You can copy this as a template:

.. code-block:: ini

   [NEAT]
   fitness_criterion     = max
   fitness_threshold     = 3.9
   pop_size              = 150
   reset_on_extinction   = False
   no_fitness_termination = False

   [DefaultGenome]
   # Network structure
   num_inputs              = 2
   num_outputs             = 1
   num_hidden              = 0
   feed_forward            = True
   initial_connection      = full
   
   # Activation function
   activation_default      = sigmoid
   activation_mutate_rate  = 0.0
   activation_options      = sigmoid
   
   # Aggregation function
   aggregation_default     = sum
   aggregation_mutate_rate = 0.0
   aggregation_options     = sum
   
   # Node bias options
   bias_init_mean          = 0.0
   bias_init_stdev         = 1.0
   bias_init_type          = gaussian
   bias_max_value          = 30.0
   bias_min_value          = -30.0
   bias_mutate_power       = 0.5
   bias_mutate_rate        = 0.7
   bias_replace_rate       = 0.1
   
   # Genome compatibility options
   compatibility_disjoint_coefficient = 1.0
   compatibility_weight_coefficient   = 0.5
   
   # Connection add/remove rates
   conn_add_prob           = 0.5
   conn_delete_prob        = 0.5
   
   # Connection enable options
   enabled_default         = True
   enabled_mutate_rate     = 0.01
   enabled_rate_to_true_add  = 0.0
   enabled_rate_to_false_add = 0.0
   
   # Node add/remove rates
   node_add_prob           = 0.2
   node_delete_prob        = 0.2
   
   # Node response options
   response_init_mean      = 1.0
   response_init_stdev     = 0.0
   response_init_type      = gaussian
   response_max_value      = 30.0
   response_min_value      = -30.0
   response_mutate_power   = 0.0
   response_mutate_rate    = 0.0
   response_replace_rate   = 0.0
   
   # Connection weight options
   weight_init_mean        = 0.0
   weight_init_stdev       = 1.0
   weight_init_type        = gaussian
   weight_max_value        = 30
   weight_min_value        = -30
   weight_mutate_power     = 0.5
   weight_mutate_rate      = 0.8
   weight_replace_rate     = 0.1
   
   # Structural mutation
   single_structural_mutation = false
   structural_mutation_surer  = default

   [DefaultSpeciesSet]
   compatibility_threshold = 3.0

   [DefaultStagnation]
   species_fitness_func = max
   max_stagnation       = 20
   species_elitism      = 2

   [DefaultReproduction]
   elitism            = 2
   survival_threshold = 0.2
   min_species_size   = 2

Annotated XOR Configuration
----------------------------

Let's walk through the XOR example config with explanations:

.. code-block:: ini

   [NEAT]
   # Use the highest fitness in the population as the criterion
   fitness_criterion     = max
   
   # Stop evolution when a genome reaches this fitness
   # For XOR: 4.0 is perfect (no error), 3.9 is "good enough"
   fitness_threshold     = 3.9
   
   # Number of genomes (organisms) in each generation
   # More = slower but better exploration. 150 is good for simple problems.
   pop_size              = 150
   
   # Don't restart from scratch if all species go extinct
   # (Rarely happens with proper configuration)
   reset_on_extinction   = False
   
   # Stop only when fitness_threshold is reached (not after fixed generations)
   no_fitness_termination = False

   [DefaultGenome]
   # Problem-specific: 2 inputs for XOR (A and B)
   num_inputs              = 2
   
   # Problem-specific: 1 output for XOR (A XOR B)
   num_outputs             = 1
   
   # Start with no hidden neurons (networks evolve complexity as needed)
   num_hidden              = 0
   
   # No recurrent connections (feed-forward only)
   # Set to False if you need memory/temporal processing
   feed_forward            = True
   
   # Start with all inputs connected to all outputs.
   # Common options:
   #   unconnected        - no initial connections
   #   full_direct        - all inputs connected to all outputs (and hidden nodes, if any)
   #   partial_direct #   - random subset of full_direct connections (0.0–1.0)
   # Legacy values ``full`` and ``partial`` are accepted for backward compatibility but
   # are deprecated; prefer the explicit variants above.
   initial_connection      = full_direct
   
   # Activation function for neurons
   # sigmoid: outputs in range (0, 1) - good for most problems
   # tanh: outputs in range (-1, 1)
   # relu: outputs in range [0, ∞)
   activation_default      = sigmoid
   
   # Probability of changing a neuron's activation function
   # 0.0 = never change (recommended to start)
   activation_mutate_rate  = 0.0
   
   # Available activation functions for mutation
   # If mutate_rate > 0, neurons can change to these functions
   activation_options      = sigmoid

   # How to combine multiple inputs to a neuron
   # sum: add all inputs (standard for neural networks)
   aggregation_default     = sum
   aggregation_mutate_rate = 0.0
   aggregation_options     = sum

   # Neuron bias (like intercept in y = mx + b)
   # Controls the neuron's firing threshold
   bias_init_mean          = 0.0      # Average starting bias
   bias_init_stdev         = 1.0      # Variation in starting bias
   bias_init_type          = gaussian # Random distribution to use
   bias_max_value          = 30.0     # Maximum allowed bias
   bias_min_value          = -30.0    # Minimum allowed bias
   bias_mutate_power       = 0.5      # How much to change when mutating
   bias_mutate_rate        = 0.7      # Probability of mutating bias
   bias_replace_rate       = 0.1      # Probability of completely replacing bias

   # How to measure genetic distance between genomes
   # Used for grouping into species
   compatibility_disjoint_coefficient = 1.0  # Weight for non-matching genes
   compatibility_weight_coefficient   = 0.5  # Weight for weight differences

   # Structural mutation probabilities (per genome, per generation)
   conn_add_prob           = 0.5      # Add a new connection
   conn_delete_prob        = 0.5      # Remove a connection
   node_add_prob           = 0.2      # Add a new hidden neuron
   node_delete_prob        = 0.2      # Remove a hidden neuron

   # Connection enable/disable options
   enabled_default         = True     # New connections start enabled
   enabled_mutate_rate     = 0.01     # Probability of toggling enabled state
   enabled_rate_to_true_add  = 0.0    # Extra probability to enable
   enabled_rate_to_false_add = 0.0    # Extra probability to disable

   # Connection weight options (like neural network weights)
   weight_init_mean        = 0.0      # Average starting weight
   weight_init_stdev       = 1.0      # Variation in starting weight
   weight_init_type        = gaussian # Random distribution
   weight_max_value        = 30       # Maximum weight magnitude
   weight_min_value        = -30      # Minimum weight magnitude
   weight_mutate_power     = 0.5      # How much to change when mutating
   weight_mutate_rate      = 0.8      # Probability of mutating weight
   weight_replace_rate     = 0.1      # Probability of completely replacing weight

   # Structural mutation control
   single_structural_mutation = false  # Allow multiple structural changes per genome
   structural_mutation_surer  = default # Fallback behavior

   [DefaultSpeciesSet]
   # How genetically similar genomes must be to belong to same species
   # Lower = more species (more diversity, slower convergence)
   # Higher = fewer species (less diversity, faster convergence)
   # 3.0 is a good starting point
   compatibility_threshold = 3.0

   [DefaultStagnation]
   # How to rank species for stagnation removal
   species_fitness_func = max  # Use best genome in species
   
   # Remove species that don't improve for this many generations
   max_stagnation       = 20
   
   # Protect the best N species from stagnation removal
   species_elitism      = 2

   [DefaultReproduction]
   # Always keep the best N genomes unchanged
   elitism            = 2
   
   # Only the top X% of each species can reproduce
   # 0.2 = top 20%
   survival_threshold = 0.2
   
   # Minimum members for a species to be maintained
   min_species_size   = 2

Key Parameters Reference
-------------------------

.. list-table:: Most Commonly Modified Parameters
   :widths: 25 15 40 20
   :header-rows: 1

   * - Parameter
     - Section
     - What It Controls
     - Typical Values
   * - ``pop_size``
     - [NEAT]
     - Population size (exploration vs speed tradeoff)
     - 50-500
   * - ``fitness_threshold``
     - [NEAT]
     - When to stop evolution
     - Problem-dependent
   * - ``num_inputs``
     - [DefaultGenome]
     - Number of input neurons
     - Problem-dependent
   * - ``num_outputs``
     - [DefaultGenome]
     - Number of output neurons
     - Problem-dependent
   * - ``activation_default``
     - [DefaultGenome]
     - Output range of neurons
     - sigmoid, tanh, relu
   * - ``feed_forward``
     - [DefaultGenome]
     - Allow recurrent connections?
     - True (most cases)
   * - ``conn_add_prob``
     - [DefaultGenome]
     - How fast networks grow
     - 0.3-0.7
   * - ``node_add_prob``
     - [DefaultGenome]
     - How fast complexity grows
     - 0.1-0.5
   * - ``compatibility_threshold``
     - [DefaultSpeciesSet]
     - Species diversity
     - 2.0-5.0
   * - ``elitism``
     - [DefaultReproduction]
     - How many top genomes to preserve
     - 1-5

Parameter Relationships
-----------------------

Understanding how parameters interact helps you tune evolution effectively:

**Population Size ↔ Generations Needed**
   Larger population explores more but needs fewer generations. Smaller population is faster per generation but needs more generations.

**Mutation Rates ↔ Network Complexity**
   Higher ``conn_add_prob`` and ``node_add_prob`` lead to more complex networks faster. If networks grow too complex, reduce these values.

**Compatibility Threshold ↔ Species Count**
   Lower threshold = more species = more diversity but slower convergence.
   Higher threshold = fewer species = faster convergence but risk of premature convergence.

**Survival Threshold ↔ Selection Pressure**
   Lower values (e.g., 0.1) = only elite can reproduce (high selection pressure).
   Higher values (e.g., 0.5) = more genomes can reproduce (low selection pressure).

**Feed Forward ↔ Problem Type**
   ``feed_forward = True``: Use for classification, function approximation, stateless problems.
   ``feed_forward = False``: Use for sequences, control with memory, temporal problems.

Common Configuration Patterns
------------------------------

Fast Convergence (When You Need Quick Results)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

   [NEAT]
   pop_size = 50                     # Smaller population
   
   [DefaultGenome]
   conn_add_prob = 0.7               # Aggressive complexification
   node_add_prob = 0.3
   
   [DefaultSpeciesSet]
   compatibility_threshold = 4.0     # Fewer species

Thorough Search (When You Want Best Solution)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

   [NEAT]
   pop_size = 300                    # Larger population
   
   [DefaultGenome]
   conn_add_prob = 0.4               # Slower complexification
   node_add_prob = 0.15
   
   [DefaultSpeciesSet]
   compatibility_threshold = 2.5     # More species

Minimal Complexity (When You Want Small Networks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

   [DefaultGenome]
   conn_add_prob = 0.3               # Rare additions
   node_add_prob = 0.1
   conn_delete_prob = 0.7            # Frequent deletions
   node_delete_prob = 0.5

Add a fitness penalty for network size in your fitness function:

.. code-block:: python

   genome.fitness = task_performance - 0.01 * len(genome.connections)

Troubleshooting
---------------

**Networks aren't learning**
   - Increase ``pop_size``
   - Decrease ``compatibility_threshold`` (more diversity)
   - Check your fitness function is rewarding progress

**Networks are too complex**
   - Decrease ``conn_add_prob`` and ``node_add_prob``
   - Increase ``conn_delete_prob`` and ``node_delete_prob``
   - Add complexity penalty to fitness

**Evolution is too slow**
   - Decrease ``pop_size``
   - Simplify your fitness function
   - Use :doc:`parallel evaluation <module_summaries>`

**All species go extinct**
   - Increase ``pop_size``
   - Increase ``compatibility_threshold``
   - Check fitness function doesn't give negative values

Next Steps
----------

**Complete Reference**
   :doc:`config_file` - Every parameter documented in detail

**Examples**
   Check ``examples/xor/config-feedforward`` in the repository for a working example

**Advanced Topics**
   :doc:`customization` - Create custom genome types, activation functions, and more

**Understanding NEAT**
   :doc:`neat_overview` - Learn how the algorithm works under the hood
