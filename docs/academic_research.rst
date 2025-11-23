NEAT-Python for Academic Research
==================================

Overview
--------

neat-python is a production-ready, well-tested implementation of the NEAT (NeuroEvolution of Augmenting Topologies) algorithm originally described by Stanley & Miikkulainen (2002). The library provides a robust foundation for neuroevolution research with comprehensive test coverage, deterministic evolution support, and pure Python implementation that facilitates modification and experimentation.

This page helps researchers understand neat-python's relationship to canonical NEAT and provides guidance for using the library in academic publications.

Strengths for Research Use
---------------------------

neat-python offers several advantages for academic research:

* **Well-tested codebase** with extensive unit test coverage and continuous integration
* **Pure Python implementation** that simplifies code inspection, modification, and extension
* **Deterministic evolution** with comprehensive seeding support for reproducible experiments
* **Parallel evaluation** capabilities for computational efficiency
* **Flexible configuration system** allowing systematic parameter exploration
* **Multiple network types** including feedforward, recurrent, CTRNN, and Izhikevich spiking networks
* **Core algorithms correctly implemented** including innovation tracking, crossover, and structural mutations

Relationship to Canonical NEAT
-------------------------------

neat-python correctly implements the most complex and critical aspects of the NEAT algorithm, particularly:

* **Innovation tracking system** with same-generation deduplication and persistence
* **Crossover mechanism** with proper gene alignment by innovation numbers
* **Structural mutations** (add-node and add-connection) following standard NEAT practice
* **75% disable rule** during crossover (with implementation note below)

However, neat-python contains several implementation choices that differ from the original 2002 NEAT paper. These design decisions that create a NEAT variant with different evolutionary dynamics. Researchers must understand these differences when publishing results.

Key Implementation Differences
-------------------------------

High-Impact Differences
^^^^^^^^^^^^^^^^^^^^^^^

**1. Fitness Sharing Mechanism**

The most significant deviation affects selection pressure and speciation dynamics.

*Canonical NEAT:*

.. code-block:: python

   adjusted_fitness = raw_fitness / species_size
   # offspring_allocation proportional to sum(adjusted_fitness per species)

*neat-python implementation:*

.. code-block:: python

   # From neat/reproduction.py
   # Species fitness is normalized to [0,1] based on population min/max
   af = (msf - min_fitness) / fitness_range

**Impact:** This creates rank-like selection pressure that scales with fitness variance. In low-variance populations, near-optimal species can be effectively eliminated. Species with many members are not penalized as in canonical NEAT, fundamentally altering the "protection of innovation" mechanism central to NEAT's design.

**Example:** If Species A has fitness 100 and Species B has fitness 101, canonical NEAT allocates offspring nearly equally (~50/50), while neat-python gives Species B 100% of offspring (minus minimums), reducing Species A to minimum size.

**2. Genomic Distance Metric**

The distance metric used for speciation differs in three ways from the canonical formula δ = c₁·E/N + c₂·D/N + c₃·W̄:

1. Node genes are included in the distance calculation (normalized by max node count)
2. Enabled/disabled state mismatch adds a +1 penalty per connection
3. Per-connection distance combines weight differences with enabled state and disjoint penalties

*Location:* ``neat/genome.py`` lines 520-561, ``neat/genes.py`` lines 151-155

**Impact:** Speciation boundaries differ from canonical NEAT, structural mutations may be penalized twice, and cluster dynamics differ even with identical hyperparameters. Direct comparison to published NEAT results becomes problematic.

**3. Disable Rule Implementation**

The 75% disable rule is applied AFTER random attribute inheritance rather than INSTEAD of it, resulting in an effective disable rate of approximately 87.5% when one parent has the gene disabled:

.. code-block:: python

   # Probability analysis:
   # Random inheritance: 50% chance of inheriting disabled state
   # Then 75% chance of disabling: 0.5 + (0.5 × 0.75) = 0.875

**Impact:** This reduces re-enabling of previously disabled connections compared to canonical NEAT and may affect network growth patterns and structural evolution.

Medium-Impact Differences
^^^^^^^^^^^^^^^^^^^^^^^^^^

**4. Node Gene Evolution**

Nodes are treated as rich, mutable structures with:

* Bias (evolves continuously)
* Response multiplier (evolves continuously)  
* Activation function (can mutate between sigmoid, tanh, ReLU, etc.)
* Aggregation function (can mutate between sum, product, max, min, etc.)

The original NEAT paper assumed fixed sigmoid activation and summation. This significantly expands the search space and changes evolutionary dynamics.

**5. Speciation Clustering**

neat-python uses "best-fit" clustering (place genome in species with closest representative) rather than the original "first-fit" approach (place in first compatible species). This is generally considered an improvement for stability and is less order-dependent.

**6. Static Compatibility Threshold**

The compatibility threshold for speciation is fixed rather than dynamically adjusted to maintain target species counts. This can cause species counts to drift over time.

**7. Initial Connectivity**

Multiple initialization topologies are available (``unconnected``, ``fs_neat_nohidden``, ``fs_neat``, ``full``, ``partial``, etc.). Canonical NEAT starts with minimal structure. Initial topology significantly affects convergence and solution quality.

**8. Spawn Allocation Smoothing**

Offspring counts are smoothed relative to previous species sizes with exact matching routines enforcing per-species minimums. This improves stability in practice but deviates from the paper.

Recommendations for Academic Use
---------------------------------

For Strict NEAT Replication Studies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your research requires exact canonical NEAT behavior for comparison with other implementations or replication of published results:

**1. Modify fitness sharing in reproduction logic**

The ``neat/reproduction.py`` module (lines 176-225) needs modification to implement canonical fitness sharing without normalization.

**2. Modify the distance metric**

Adjust ``neat/genome.py`` to match the canonical formula:

* Remove node gene contributions
* Remove enabled/disabled penalty
* Compute W̄ as pure average weight difference on matching connections
* Expose c₁, c₂, c₃ as separate configuration parameters

**3. Configure for canonical search space**

Lock configuration to canonical NEAT parameters:

.. code-block:: ini

   [DefaultGenome]
   activation_mutate_rate = 0.0
   activation_options = sigmoid
   aggregation_mutate_rate = 0.0
   aggregation_options = sum
   response_mutate_rate = 0.0
   initial_connection = unconnected
   feed_forward = True

**4. Consider implementing dynamic compatibility threshold**

For species count stabilization, add threshold adjustment mechanisms.

**5. Run sensitivity analysis**

Evaluate the impact of the disable rate difference (75% vs 87.5%).

For General NEAT-Variant Research
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The library is well-suited for research involving NEAT variants or algorithmic modifications. In this case:

**1. Document all configuration parameters** in your methodology section

**2. Explicitly note implementation differences** from canonical NEAT

**3. Describe your method as "NEAT-inspired" or "NEAT-variant"** rather than pure NEAT

**4. Run multiple trials** with different random seeds for statistical validity

**5. Report statistical measures** (mean, standard deviation, confidence intervals)

**6. Compare within-implementation** rather than across different NEAT implementations

General Best Practices
^^^^^^^^^^^^^^^^^^^^^^^

For both strict replication and variant research:

**1. Document version number**

Always specify which version of neat-python you used. APIs and behavior may change between versions.

**2. Provide configuration files**

Include your complete configuration files in supplementary materials or code repositories.

**3. Seed all RNGs**

For reproducibility, seed not only neat-python (via ``Population(seed=...)``), but also NumPy, PyTorch, and any other libraries used in your fitness evaluation function.

**4. Be transparent about implementation**

Clearly state that you are using neat-python rather than the original C++ implementation.

**5. Consider validation experiments**

If making comparisons across implementations, run validation experiments on simple problems with known characteristics.

**6. Archive complete code**

Provide complete, runnable code including the specific neat-python version and any modifications you made.

Configuration for Canonical Behavior
-------------------------------------

If you need behavior closer to canonical NEAT without code modification, use this configuration template as a starting point:

.. code-block:: ini

   [NEAT]
   fitness_criterion     = max
   fitness_threshold     = <your threshold>
   pop_size              = 150
   reset_on_extinction   = False

   [DefaultGenome]
   # Lock to canonical activation/aggregation
   activation_default      = sigmoid
   activation_mutate_rate  = 0.0
   activation_options      = sigmoid
   
   aggregation_default     = sum
   aggregation_mutate_rate = 0.0
   aggregation_options     = sum
   
   # Disable response evolution
   response_mutate_rate    = 0.0
   response_replace_rate   = 0.0
   
   # Minimal initial connectivity
   initial_connection      = unconnected
   
   # Standard NEAT topology
   feed_forward            = True
   
   # Configure other mutation rates as needed for your problem
   bias_mutate_rate        = 0.7
   bias_replace_rate       = 0.1
   conn_add_prob           = 0.5
   conn_delete_prob        = 0.5
   node_add_prob           = 0.2
   node_delete_prob        = 0.2
   weight_mutate_rate      = 0.8
   weight_replace_rate     = 0.1

   [DefaultSpeciesSet]
   compatibility_threshold = 3.0

   [DefaultStagnation]
   species_fitness_func = max
   max_stagnation       = 20
   species_elitism      = 2

   [DefaultReproduction]
   elitism            = 2
   survival_threshold = 0.2

Note that even with this configuration, fitness sharing and distance metric differences remain. Complete canonical behavior requires code modification.

Working with Deterministic Evolution
-------------------------------------

neat-python supports fully deterministic evolution for reproducible experiments:

.. code-block:: python

   import random
   import neat

   # Seed the standard library random module
   random.seed(42)
   
   # Create population with seed
   config = neat.Config(...)
   population = neat.Population(config, seed=42)
   
   # For parallel evaluation, also seed the evaluator
   with neat.ParallelEvaluator(4, eval_genome, seed=42) as pe:
       winner = population.run(pe.evaluate, 100)

**Important:** You must also seed any RNGs used in your evaluation function (NumPy, PyTorch, TensorFlow, etc.) for complete reproducibility.

See :doc:`reproducibility` for comprehensive guidance.

Publication Guidelines
----------------------

When publishing research using neat-python:

Methodology Section
^^^^^^^^^^^^^^^^^^^

Include the following information:

1. **Implementation**: "We used neat-python version X.Y.Z (Stanley et al., 2015), a Python implementation of NEAT."

2. **Deviations** (if relevant): "neat-python implements normalized fitness sharing and an extended distance metric that differ from canonical NEAT [cite canonical paper]. See supplementary materials for complete configuration."

3. **Configuration**: "All configuration parameters are provided in supplementary materials. Key parameters were: population size N, compatibility threshold δ, ..."

4. **Reproducibility**: "All experiments used fixed random seeds for reproducibility. Complete code is available at [repository URL]."

Results Section
^^^^^^^^^^^^^^^

1. **Multiple trials**: Report statistics from multiple independent runs (typically 20-50 runs)

2. **Statistical measures**: Include mean, standard deviation, and confidence intervals

3. **Significance testing**: When comparing methods, use appropriate statistical tests

4. **Variability**: Discuss and visualize variability across runs

Discussion Section
^^^^^^^^^^^^^^^^^^

If your results differ from published NEAT results using other implementations, acknowledge that implementation differences may contribute to observed variations.

Comparison Studies
^^^^^^^^^^^^^^^^^^

When comparing neat-python results to other NEAT implementations or algorithms:

1. **Acknowledge implementation differences**: "While both implement the NEAT algorithm, implementation differences in fitness sharing and speciation may affect results."

2. **Control what you can**: Use similar configuration parameters, population sizes, and evaluation budgets

3. **Focus on trends**: Compare relative performance trends rather than absolute values

4. **Consider validation**: Include simple validation problems where expected behavior is well-understood

References Section
^^^^^^^^^^^^^^^^^^

Include appropriate citations:

* **Original NEAT paper**: Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. *Evolutionary Computation*, 10(2), 99-127.

* **neat-python**: Reference the GitHub repository and/or documentation: https://github.com/CodeReclaimers/neat-python

Additional Considerations
-------------------------

CTRNN and Specialized Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using CTRNN or other specialized network types:

* **Integration method**: neat-python uses explicit Euler integration with fixed time steps
* **Numerical stability**: Document ``dt`` and ``time_constant`` values; consider sensitivity analysis
* **Validation**: For dynamics-sensitive applications, validate numerical behavior on simple test cases

See :doc:`ctrnn` for implementation details.

Checkpoint Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^

Version 1.0 introduced breaking changes to implement proper innovation tracking. Checkpoints from earlier versions are not compatible. Always specify the version used when archiving research code.

Extension and Customization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

neat-python's pure Python implementation facilitates algorithmic research. You can:

* Implement custom activation functions
* Modify reproduction operators
* Create custom speciation methods
* Add new mutation operators
* Implement alternative selection schemes

See :doc:`customization` for guidance on extending the library.

Summary
-------

neat-python is a robust, well-maintained NEAT implementation suitable for serious academic research when used with awareness of its characteristics. The library correctly implements the most critical algorithmic components (innovation tracking, crossover, structural mutations) but differs from canonical NEAT in fitness sharing and distance calculation.

**For canonical NEAT replication:** Code modifications and careful configuration are necessary.

**For NEAT-variant research:** The library is excellent as-is, provided implementation choices are documented in publications.

**For all research:** Understanding these differences enables informed use and appropriate interpretation of results. The pure Python implementation and comprehensive test suite make verification and modification straightforward—a significant advantage for academic research.

References
----------

Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. *Evolutionary Computation*, 10(2), 99-127.

Stanley, K. O., & Miikkulainen, R. (2004). Competitive coevolution through evolutionary complexification. *Journal of Artificial Intelligence Research*, 21, 63-100.

Further Reading
---------------

* :doc:`neat_overview` - High-level overview of NEAT concepts
* :doc:`innovation_numbers` - Deep dive into innovation tracking implementation
* :doc:`reproducibility` - Complete guide to deterministic evolution
* :doc:`config_file` - Detailed configuration parameter reference
* :doc:`customization` - Guide to extending and modifying neat-python
