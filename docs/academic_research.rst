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

However, neat-python contains several implementation choices that differ from the original 2002 NEAT paper. Many of these differences are now configurable, allowing researchers to choose between neat-python's default behavior and canonical NEAT behavior.

Key Implementation Differences
-------------------------------

High-Impact Differences
^^^^^^^^^^^^^^^^^^^^^^^

**1. Fitness Sharing Mechanism**

The default behavior uses normalized fitness sharing, which differs from canonical NEAT.

*Canonical NEAT:*

.. code-block:: python

   adjusted_fitness = raw_fitness / species_size
   # offspring_allocation proportional to sum(adjusted_fitness per species)

*neat-python default (normalized):*

.. code-block:: python

   af = (msf - min_fitness) / fitness_range  # Normalized to [0, 1]

**Impact:** Normalized sharing creates rank-like selection pressure. In low-variance populations, near-optimal species can be effectively eliminated.

**Configurable:** Set ``fitness_sharing = canonical`` in ``[DefaultReproduction]`` for paper-faithful behavior. Default ``normalized`` preserves existing behavior.

**2. Genomic Distance Metric**

The distance metric used for speciation is now configurable to match the canonical formula δ = c₁·E/N + c₂·D/N + c₃·W̄.

Connection genes are matched by **innovation number** (consistent with crossover). The full distance formula is:

.. math::

   \delta = \delta_{\text{nodes}} + \delta_{\text{connections}}

**Connection gene distance:**

.. math::

   \delta_{\text{connections}} = \frac{c_2 \cdot D + c_1 \cdot E + \sum_{i \in M} d_i}{N_c}

Where:

- *D* = number of disjoint connection genes (innovation within the other genome's range)
- *E* = number of excess connection genes (innovation beyond the other genome's range)
- *M* = set of homologous (matching) connection gene pairs
- *d_i* = (|w₁ - w₂| + p · [e₁ ≠ e₂]) · c₃ for each matching pair, where *p* is ``compatibility_enable_penalty`` (default 1.0)
- *N_c* = max number of connection genes in either genome
- *c₁* = ``compatibility_excess_coefficient`` (defaults to ``compatibility_disjoint_coefficient`` if set to ``auto``)
- *c₂* = ``compatibility_disjoint_coefficient``
- *c₃* = ``compatibility_weight_coefficient``

**Node gene distance** (when ``compatibility_include_node_genes`` is True, the default):

.. math::

   \delta_{\text{nodes}} = \frac{c_2 \cdot D_n + \sum_{i \in M_n} d_i^{(n)}}{N_n}

Where node distance *d_i^(n)* includes bias difference, response difference, time_constant difference, and +1 penalties for activation and aggregation function mismatches, all scaled by c₃.

**To approximate canonical NEAT distance:**

- Set ``compatibility_include_node_genes = False``
- Set ``compatibility_enable_penalty = 0.0``
- Set ``compatibility_excess_coefficient`` explicitly if c₁ ≠ c₂

**3. Disable Rule Implementation**

The 75% disable rule now correctly matches the paper: when either parent has a gene disabled, the offspring's enabled state is determined by a fresh 75/25 coin flip (75% disabled, 25% enabled), replacing whatever value was randomly inherited.

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

**6. Dynamic Compatibility Threshold**

By default the compatibility threshold for speciation is fixed. Set ``target_num_species`` in ``[DefaultSpeciesSet]`` to enable dynamic adjustment toward a target species count (as described in the paper). Configure ``threshold_adjust_rate``, ``threshold_min``, and ``threshold_max`` to control the adjustment behavior.

**7. Initial Connectivity**

Multiple initialization topologies are available (``unconnected``, ``fs_neat_nohidden``, ``fs_neat``, ``full``, ``partial``, etc.). Canonical NEAT starts with minimal structure. Initial topology significantly affects convergence and solution quality.

**8. Spawn Allocation Smoothing**

By default, offspring counts are smoothed relative to previous species sizes. Set ``spawn_method = proportional`` in ``[DefaultReproduction]`` for direct proportional allocation (canonical NEAT). Default ``smoothed`` preserves existing behavior.

Recommendations for Academic Use
---------------------------------

For Strict NEAT Replication Studies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your research requires exact canonical NEAT behavior for comparison with other implementations or replication of published results, all major differences are now configurable:

**1. Use canonical fitness sharing and spawn allocation**

Set ``fitness_sharing = canonical`` and ``spawn_method = proportional`` in ``[DefaultReproduction]``.

**2. Configure the distance metric for canonical NEAT**

Set ``compatibility_include_node_genes = False`` and ``compatibility_enable_penalty = 0.0`` in ``[DefaultGenome]``. Optionally set ``compatibility_excess_coefficient`` if c₁ ≠ c₂.

**3. Enable dynamic compatibility threshold**

Set ``target_num_species`` in ``[DefaultSpeciesSet]`` to maintain a target species count.

**4. Enable interspecies crossover**

Set ``interspecies_crossover_prob`` to a small value (e.g., 0.001) in ``[DefaultReproduction]``.

**5. Lock activation and aggregation to canonical defaults**

.. code-block:: ini

   [DefaultGenome]
   activation_mutate_rate = 0.0
   activation_options = sigmoid
   aggregation_mutate_rate = 0.0
   aggregation_options = sum
   response_mutate_rate = 0.0
   initial_connection = unconnected
   feed_forward = True

See the "Configuration for Canonical Behavior" section below for a complete template.

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

neat-python now supports canonical NEAT behavior through configuration alone. Use this template as a starting point:

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

   # Canonical distance metric: no node genes, no enable penalty
   compatibility_include_node_genes = False
   compatibility_enable_penalty     = 0.0

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
   target_num_species      = 10

   [DefaultStagnation]
   species_fitness_func = max
   max_stagnation       = 20
   species_elitism      = 2

   [DefaultReproduction]
   elitism                     = 2
   survival_threshold          = 0.2
   fitness_sharing             = canonical
   spawn_method                = proportional
   interspecies_crossover_prob = 0.001

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

* **neat-python**: McIntyre, A., Kallada, M., Miguel, C. G., Feher de Silva, C., & Netto, M. L. *neat-python* (Version 2.0.1) [Computer software]. https://doi.org/10.5281/zenodo.19024753

Additional Considerations
-------------------------

CTRNN and Specialized Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using CTRNN or other specialized network types:

* **Integration method**: neat-python uses exponential Euler (ETD1) integration, which exactly integrates the linear decay term and is unconditionally stable regardless of ``dt/tau``
* **Numerical stability**: Document ``dt`` and ``time_constant`` values; the exponential Euler method eliminates the ``dt < 2*tau`` stability constraint of forward Euler
* **Validation**: For dynamics-sensitive applications, validate numerical behavior on simple test cases
* **GPU acceleration**: For large populations, optional GPU-accelerated evaluation is available via ``neat.gpu`` (requires CuPy). The GPU evaluator uses the same integration method as the CPU implementation.

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

neat-python is a robust, well-maintained NEAT implementation suitable for serious academic research. The library correctly implements the critical algorithmic components (innovation tracking, crossover, structural mutations) and now offers configuration options for canonical NEAT behavior.

**For canonical NEAT replication:** All major differences from the paper are configurable without code modification. See the configuration template above.

**For NEAT-variant research:** The library is excellent with default settings, provided implementation choices are documented in publications.

**For all research:** The pure Python implementation and comprehensive test suite make verification and modification straightforward—a significant advantage for academic research.

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
