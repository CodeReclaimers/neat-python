Migration Guide for neat-python 1.0
====================================

This guide helps you migrate from neat-python 0.93 to 1.0, which includes breaking changes to the parallel evaluation APIs.

Overview of Changes
-------------------

Removed Components
~~~~~~~~~~~~~~~~~~

- **ThreadedEvaluator** - Removed due to minimal utility (Python GIL) and implementation issues
- **DistributedEvaluator** - Removed due to instability and complexity

Improved Components
~~~~~~~~~~~~~~~~~~~

- **ParallelEvaluator** - Now supports context manager protocol for proper resource cleanup

ThreadedEvaluator (Removed)
----------------------------

Why Was It Removed?
~~~~~~~~~~~~~~~~~~~

The ``ThreadedEvaluator`` provided minimal benefit for most use cases:

- Python's Global Interpreter Lock (GIL) prevents true parallel execution of CPU-bound code
- Only beneficial for I/O-bound fitness functions (rare in neural network evolution)
- Had implementation issues including unreliable cleanup and potential deadlocks
- No timeout on output queue operations could cause indefinite hangs

Migration Path
~~~~~~~~~~~~~~

**For CPU-bound fitness evaluation (most common):**

Use ``ParallelEvaluator`` instead, which uses process-based parallelism to bypass the GIL:

.. code-block:: python

    # Old code (ThreadedEvaluator)
    import neat

    evaluator = neat.ThreadedEvaluator(4, eval_genome)
    winner = population.run(evaluator.evaluate, 300)
    evaluator.stop()  # Manual cleanup

    # New code (ParallelEvaluator with context manager)
    import neat
    import multiprocessing

    with neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome) as evaluator:
        winner = population.run(evaluator.evaluate, 300)
    # Automatic cleanup on context exit

**For I/O-bound fitness evaluation (uncommon):**

Consider using Python's ``asyncio`` for truly I/O-bound operations, or still use ``ParallelEvaluator`` which works well for both CPU and I/O-bound tasks.

DistributedEvaluator (Removed)
-------------------------------

Why Was It Removed?
~~~~~~~~~~~~~~~~~~~

The ``DistributedEvaluator`` had several fundamental problems:

- Marked as **beta/unstable** in the documentation since its introduction
- Used ``multiprocessing.managers`` which is notoriously unreliable across networks
- Integration tests were skipped due to pickling and reliability issues
- 574 lines of complex, fragile code with extensive error handling
- Better alternatives exist for distributed computing

Migration Path
~~~~~~~~~~~~~~

Option 1: Single-machine parallelism (simplest)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you were using ``DistributedEvaluator`` on a single machine, migrate to ``ParallelEvaluator``:

.. code-block:: python

    # Old code (DistributedEvaluator - single machine)
    import neat

    de = neat.DistributedEvaluator(
        ('localhost', 8022),
        authkey=b'password',
        eval_function=eval_genome,
        mode=neat.distributed.MODE_PRIMARY
    )
    de.start()
    winner = population.run(de.evaluate, 300)
    de.stop()

    # New code (ParallelEvaluator)
    import neat
    import multiprocessing

    with neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome) as evaluator:
        winner = population.run(evaluator.evaluate, 300)

Option 2: Multi-machine distributed computing (recommended for large-scale)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use established distributed computing frameworks like **Ray** or **Dask**.

Using Ray (recommended)
"""""""""""""""""""""""

.. code-block:: python

    import neat
    import ray

    # Initialize Ray
    ray.init(address='auto')  # or ray.init() for local cluster

    @ray.remote
    def eval_genome_remote(genome, config):
        """Fitness evaluation function wrapped for Ray."""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # Your fitness evaluation logic here
        return fitness_value

    def eval_genomes_distributed(genomes, config):
        """Fitness function that distributes work via Ray."""
        # Submit all evaluation tasks
        futures = [eval_genome_remote.remote(genome, config)
                   for genome_id, genome in genomes]

        # Gather results
        results = ray.get(futures)

        # Assign fitness values
        for (genome_id, genome), fitness in zip(genomes, results):
            genome.fitness = fitness

    # Use with NEAT
    population = neat.Population(config)
    winner = population.run(eval_genomes_distributed, 300)

Using Dask
""""""""""

.. code-block:: python

    import neat
    from dask.distributed import Client

    # Connect to Dask cluster
    client = Client('scheduler-address:8786')

    def eval_genome_dask(genome, config):
        """Fitness evaluation function."""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # Your fitness evaluation logic here
        return fitness_value

    def eval_genomes_distributed(genomes, config):
        """Fitness function that distributes work via Dask."""
        # Submit all evaluation tasks
        futures = [client.submit(eval_genome_dask, genome, config)
                   for genome_id, genome in genomes]

        # Gather results
        results = client.gather(futures)

        # Assign fitness values
        for (genome_id, genome), fitness in zip(genomes, results):
            genome.fitness = fitness

    # Use with NEAT
    population = neat.Population(config)
    winner = population.run(eval_genomes_distributed, 300)

Option 3: Custom solution
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can implement your own distributed evaluation using:

- Message queues (RabbitMQ, Redis, AWS SQS)
- Task queues (Celery)
- Cloud functions (AWS Lambda, Google Cloud Functions)

ParallelEvaluator Improvements
-------------------------------

The ``ParallelEvaluator`` has been improved with proper resource management and context manager support.

Context Manager Pattern (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**New recommended usage:**

.. code-block:: python

    import neat
    import multiprocessing

    with neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome) as evaluator:
        winner = population.run(evaluator.evaluate, 300)
    # Pool is automatically cleaned up when exiting the context

**Benefits:**

- Guaranteed cleanup of multiprocessing pool
- No risk of zombie processes
- Cleaner, more Pythonic code
- Exception-safe resource management

Backward Compatibility
~~~~~~~~~~~~~~~~~~~~~~

**Old usage still works:**

.. code-block:: python

    import neat
    import multiprocessing

    evaluator = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = population.run(evaluator.evaluate, 300)
    # Pool will be cleaned up by __del__, but context manager is preferred

While the old pattern still functions, we **strongly recommend** migrating to the context manager pattern for better resource management.

Explicit Cleanup
~~~~~~~~~~~~~~~~

If you need explicit control over cleanup:

.. code-block:: python

    evaluator = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    try:
        winner = population.run(evaluator.evaluate, 300)
    finally:
        evaluator.close()  # Explicit cleanup

Additional Resources
--------------------

- **Ray Documentation**: https://docs.ray.io/
- **Dask Documentation**: https://docs.dask.org/
- **neat-python Documentation**: http://neat-python.readthedocs.io/
- **GitHub Repository**: https://github.com/CodeReclaimers/neat-python

Getting Help
------------

If you encounter issues during migration:

1. Check the `GitHub Issues <https://github.com/CodeReclaimers/neat-python/issues>`_ for similar problems
2. Review the updated `documentation <http://neat-python.readthedocs.io/>`_
3. Open a new issue with details about your migration challenge

Version Information
-------------------

- This guide applies to migration from neat-python 0.93 → 1.0
- Last updated: 2025-11-09

Configuration File Migration (v1.0)
====================================

Starting with v1.0, neat-python requires **all configuration parameters to be explicitly specified** in your configuration file. This section guides you through updating your configuration files for compatibility with v1.0.

What Changed
------------

Previously, if you omitted certain parameters, neat-python would use default values and issue deprecation warnings (which were easy to miss). Now, **missing required parameters cause immediate errors** with helpful suggestions on what to add.

This change improves reliability and reproducibility:

- **No silent defaults**: You always know exactly what configuration you're using
- **Explicit is better than implicit**: Your config file is now self-documenting
- **Prevents mistakes**: Can't accidentally run experiments with unintended default values
- **Better for v1.0**: Breaking change is appropriate for a major version

What You'll See
---------------

If you're missing a required parameter, you'll get a clear error message like this:

.. code-block:: text

    RuntimeError: Missing required configuration item: 'bias_init_type'
    This parameter must be explicitly specified in your configuration file.
    Suggested value: bias_init_type = gaussian

The error tells you:

1. Exactly which parameter is missing
2. That it must be added to your config file
3. A suggested value to use

Required Parameters by Section
-------------------------------

[NEAT] Section
~~~~~~~~~~~~~~

Add this parameter if missing:

.. code-block:: ini

    no_fitness_termination = False

Set to ``True`` if you want evolution to run for a fixed number of generations regardless of fitness. Set to ``False`` to use fitness-based termination.

[DefaultGenome] Section
~~~~~~~~~~~~~~~~~~~~~~~

Add these parameters if missing:

.. code-block:: ini

    # Structural mutation parameters
    single_structural_mutation     = false
    structural_mutation_surer      = default

    # Initialization type parameters
    bias_init_type                 = gaussian
    response_init_type             = gaussian
    weight_init_type               = gaussian

    # Connection enable mutation parameters
    enabled_rate_to_true_add       = 0.0
    enabled_rate_to_false_add      = 0.0

**Parameter meanings:**

- ``single_structural_mutation``: Set to ``true`` to allow only one structural mutation (add/remove node/connection) per genome per generation
- ``structural_mutation_surer``: Controls fallback behavior when structural mutations fail (``default`` uses same value as ``single_structural_mutation``)
- ``*_init_type``: Distribution type for initializing values (``gaussian`` for normal distribution or ``uniform`` for uniform distribution)
- ``enabled_rate_to_*_add``: Additional probability to enable/disable connections during mutation

[DefaultStagnation] Section
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add this parameter if missing:

.. code-block:: ini

    species_elitism = 2

This sets the number of species protected from stagnation removal. For example, ``species_elitism = 2`` protects the 2 best-performing species from being removed due to stagnation.

[DefaultReproduction] Section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add this parameter if missing:

.. code-block:: ini

    min_species_size = 2

This sets the minimum size for a species to be maintained after reproduction.

[IZGenome] Section (Spiking Networks Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're using IZNN genomes (Izhikevich spiking neuron model), also add:

.. code-block:: ini

    a_init_type = gaussian
    b_init_type = gaussian
    c_init_type = gaussian
    d_init_type = gaussian

Complete Example Configuration
-------------------------------

Here's a minimal complete configuration file for v1.0 (feedforward network solving XOR):

.. code-block:: ini

    [NEAT]
    fitness_criterion     = max
    fitness_threshold     = 0.9
    pop_size              = 150
    reset_on_extinction   = False
    no_fitness_termination = False

    [DefaultGenome]
    # Node activation options
    activation_default      = sigmoid
    activation_mutate_rate  = 0.0
    activation_options      = sigmoid

    # Node aggregation options
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

    # Network topology
    feed_forward            = True
    initial_connection      = full

    # Node add/remove rates
    node_add_prob           = 0.2
    node_delete_prob        = 0.2

    # Network parameters
    num_hidden              = 0
    num_inputs              = 2
    num_outputs             = 1

    # Node response options
    response_init_mean      = 1.0
    response_init_stdev     = 0.0
    response_init_type      = gaussian
    response_max_value      = 30.0
    response_min_value      = -30.0
    response_mutate_power   = 0.0
    response_mutate_rate    = 0.0
    response_replace_rate   = 0.0

    # Structural mutation
    single_structural_mutation  = false
    structural_mutation_surer   = default

    # Connection weight options
    weight_init_mean        = 0.0
    weight_init_stdev       = 1.0
    weight_init_type        = gaussian
    weight_max_value        = 30
    weight_min_value        = -30
    weight_mutate_power     = 0.5
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
    min_species_size   = 2

Step-by-Step Migration Instructions
------------------------------------

1. **Run your existing code** - If you have missing parameters, you'll get a clear error message
2. **Add the suggested parameter** - Copy the suggested line from the error message into your config file under the appropriate section
3. **Repeat** - Continue running and adding parameters until no errors remain
4. **Verify** - Run your experiments to ensure everything works as expected

.. note::
    The error messages are designed to be helpful and actionable. They tell you exactly what to add and where.

Where to Find Example Configs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All example configurations in the ``examples/`` directory have been updated for v1.0. You can use these as references:

- ``examples/xor/config-feedforward`` - Simple feedforward network
- ``examples/xor/config-feedforward-partial`` - Partial connectivity
- ``examples/xor/config-spiking`` - Spiking neural network (IZNN)

Quick Reference: New Required Parameters
-----------------------------------------

Here's a quick summary of all parameters that became required in v1.0:

**[NEAT] section:**

- ``no_fitness_termination`` - Controls whether fitness-based termination is used

**[DefaultGenome] section:**

- ``single_structural_mutation`` - Limit to one structural mutation per generation
- ``structural_mutation_surer`` - Fallback behavior for failed mutations
- ``bias_init_type`` - Distribution for bias initialization
- ``response_init_type`` - Distribution for response initialization
- ``weight_init_type`` - Distribution for weight initialization
- ``enabled_rate_to_true_add`` - Extra probability to enable connections
- ``enabled_rate_to_false_add`` - Extra probability to disable connections

**[DefaultStagnation] section:**

- ``species_elitism`` - Number of species protected from stagnation removal

**[DefaultReproduction] section:**

- ``min_species_size`` - Minimum size for species maintenance

**[IZGenome] section (if using spiking networks):**

- ``a_init_type``, ``b_init_type``, ``c_init_type``, ``d_init_type`` - Distributions for Izhikevich parameters

For detailed documentation on what each parameter does, see :doc:`config_file`.

Configuration Migration Resources
----------------------------------

- **Configuration File Reference**: :doc:`config_file`
- **Example Configurations**: ``examples/`` directory in the repository
- **GitHub Issues**: https://github.com/CodeReclaimers/neat-python/issues
- **Documentation**: http://neat-python.readthedocs.io/
