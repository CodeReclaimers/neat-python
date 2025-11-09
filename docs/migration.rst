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
