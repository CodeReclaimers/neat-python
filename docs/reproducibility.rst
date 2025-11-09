.. _reproducibility-label:

Reproducibility
===============

Overview
--------

Reproducibility is essential for evolutionary algorithms like NEAT in several contexts:

* **Debugging**: Reproduce exact behavior to isolate and fix issues
* **Scientific Research**: Enable others to verify and build upon your results
* **Algorithm Comparison**: Fair evaluation requires consistent random behavior
* **Development**: Test changes without random variation masking bugs

By default, NEAT-Python uses Python's ``random`` module for all stochastic operations (initialization, mutation, crossover, parent selection). Without setting a seed, each run produces different results due to random initialization of Python's random number generator.

Basic Usage
-----------

Setting a Seed via Config File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to enable reproducibility is to add a ``seed`` parameter to your configuration file:

.. code-block:: ini

   [NEAT]
   fitness_criterion = max
   fitness_threshold = 100.0
   pop_size = 150
   reset_on_extinction = False
   no_fitness_termination = False
   seed = 42  # Enable reproducibility

With this configuration, every run will produce identical evolution trajectories (assuming your fitness function is also deterministic or seeded separately).

Setting a Seed via Population Parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also set the seed programmatically when creating a Population:

.. code-block:: python

   import neat
   
   config = neat.Config(neat.DefaultGenome,
                       neat.DefaultReproduction,
                       neat.DefaultSpeciesSet,
                       neat.DefaultStagnation,
                       'config-file')
   
   # With reproducibility
   pop = neat.Population(config, seed=42)
   winner = pop.run(eval_genomes, 100)

The ``seed`` parameter takes precedence over the config file setting:

.. code-block:: python

   # Config file has seed=100, but this uses seed=42
   pop = neat.Population(config, seed=42)

Random Behavior (Default)
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you omit the seed or set it to ``None``, NEAT-Python behaves non-deterministically:

.. code-block:: python

   # No seed - different results each run
   pop = neat.Population(config)
   winner = pop.run(eval_genomes, 100)

This is the default behavior and is suitable for final testing where you want to evaluate robustness across multiple random initializations.

Parallel Mode
-------------

Parallel evaluation using ``ParallelEvaluator`` also supports reproducibility with a per-genome seeding strategy.

Basic Parallel Reproducibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import multiprocessing
   import neat
   
   def eval_genome(genome, config):
       """Fitness function that may use random numbers."""
       net = neat.nn.FeedForwardNetwork.create(genome, config)
       # Your fitness evaluation here
       # Can use random.random(), random.choice(), etc.
       return fitness
   
   config = neat.Config(...)
   
   # Create parallel evaluator with seed
   with neat.ParallelEvaluator(multiprocessing.cpu_count(),
                                eval_genome,
                                seed=42) as evaluator:
       pop = neat.Population(config, seed=42)
       winner = pop.run(evaluator.evaluate, 100)

How Parallel Seeding Works
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ParallelEvaluator`` uses a deterministic per-genome seeding strategy:

* Each genome gets a unique seed: ``base_seed + genome.key``
* Same genome always gets the same random sequence (given same base seed)
* Different genomes get different but reproducible random sequences
* Results are reproducible across runs and number of worker processes

This ensures that:

1. **Reproducibility**: Same seed produces identical results
2. **Independence**: Different genomes get different random numbers
3. **Determinism**: Same genome evaluated multiple times gets same fitness

Example: Each genome with key 1, 2, 3, ... gets seeds 43, 44, 45, ... (if base seed is 42).

Parallel Limitations
^^^^^^^^^^^^^^^^^^^^

The parallel seeding strategy only works when:

* Your fitness function uses Python's ``random`` module
* Fitness evaluation is deterministic given the random seed
* No external non-deterministic factors (network, hardware timing, etc.)

Checkpointing
-------------

The checkpoint system automatically preserves and restores random state, ensuring perfect reproducibility when resuming from checkpoints.

Saving Checkpoints
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import neat
   
   config = neat.Config(...)
   pop = neat.Population(config, seed=42)
   
   # Save checkpoint every 5 generations
   pop.add_reporter(neat.Checkpointer(5))
   
   # Run evolution
   pop.run(eval_genomes, 50)

Restoring Checkpoints
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Restore from checkpoint
   pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-25')
   
   # Continue evolution - will produce same results as if uninterrupted
   pop.run(eval_genomes, 50)

The restored population continues with the exact random state from the checkpoint, ensuring the continued evolution is identical to what it would have been without interruption.

Checkpoint Reproducibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Checkpoints preserve:

* Population state (all genomes and their attributes)
* Species structure
* Generation counter  
* **Random number generator state**
* Innovation tracker state

This means checkpointing maintains reproducibility even without explicitly setting a seed when restoring.

Limitations
-----------

Python Random Module Only
^^^^^^^^^^^^^^^^^^^^^^^^^^

NEAT-Python's seed parameter only controls Python's ``random`` module. If your fitness function uses other random number generators, you must seed them separately.

**NumPy Example:**

.. code-block:: python

   import random
   import numpy as np
   import neat
   
   # Seed all RNG sources
   random.seed(42)
   np.random.seed(42)
   
   pop = neat.Population(config, seed=42)
   winner = pop.run(eval_genomes, 100)

**PyTorch Example:**

.. code-block:: python

   import random
   import torch
   import neat
   
   # Seed all RNG sources
   random.seed(42)
   torch.manual_seed(42)
   
   pop = neat.Population(config, seed=42)
   winner = pop.run(eval_genomes, 100)

Non-Deterministic Fitness Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some fitness evaluations are inherently non-deterministic:

* **External simulators** with their own RNG
* **Network communication** with variable latency
* **Hardware timing** dependencies
* **File system operations** with non-deterministic ordering
* **Multi-threaded code** with race conditions

In these cases, reproducibility may not be achievable even with proper seeding. Consider:

1. Using deterministic simulation modes if available
2. Mocking non-deterministic components during testing
3. Averaging over multiple evaluations to reduce variance
4. Documenting the non-deterministic sources in your results

Python Version Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Random number sequences may differ between Python versions or implementations (CPython vs PyPy). For complete reproducibility:

* Document the Python version used
* Use the same Python version for reproduction
* Be aware that upgrading Python might change random sequences

Best Practices
--------------

When to Use Seeds
^^^^^^^^^^^^^^^^^

**Development and Debugging:**

* Use fixed seed during development
* Makes behavior predictable and repeatable
* Easier to debug issues and compare changes

.. code-block:: python

   # Development: use fixed seed
   pop = neat.Population(config, seed=42)

**Scientific Research:**

* Use fixed seed for fair algorithm comparison
* Document seed in publications and code
* Enables others to reproduce your exact results

.. code-block:: python

   # Research: fixed seed, document in paper
   SEED = 42  # Seed used for all experiments
   pop = neat.Population(config, seed=SEED)

**Production/Final Testing:**

* Run with multiple different seeds
* Report statistics (mean, std, min, max) across runs
* Ensures results generalize beyond single random initialization

.. code-block:: python

   # Production: multiple seeds for robustness
   results = []
   for seed in [42, 123, 456, 789, 999]:
       pop = neat.Population(config, seed=seed)
       winner = pop.run(eval_genomes, 100)
       results.append(winner.fitness)
   
   print(f"Mean fitness: {np.mean(results):.2f}")
   print(f"Std dev: {np.std(results):.2f}")

Recommended Workflow
^^^^^^^^^^^^^^^^^^^^

1. **Develop with fixed seed** - Find bugs and tune parameters

.. code-block:: python

   # Step 1: Development
   pop = neat.Population(config, seed=42)
   winner = pop.run(eval_genomes, 100)
   print(f"Winner fitness: {winner.fitness}")

2. **Verify reproducibility** - Confirm seed works correctly

.. code-block:: python

   # Step 2: Verify reproducibility
   pop1 = neat.Population(config, seed=42)
   winner1 = pop1.run(eval_genomes, 100)
   
   pop2 = neat.Population(config, seed=42)
   winner2 = pop2.run(eval_genomes, 100)
   
   assert winner1.fitness == winner2.fitness
   print("✓ Results are reproducible!")

3. **Evaluate with multiple seeds** - Test robustness

.. code-block:: python

   # Step 3: Multiple seeds for final evaluation
   import numpy as np
   
   results = []
   seeds = [42, 123, 456, 789, 999, 111, 222, 333, 444, 555]
   
   for seed in seeds:
       pop = neat.Population(config, seed=seed)
       winner = pop.run(eval_genomes, 100)
       results.append({
           'seed': seed,
           'fitness': winner.fitness,
           'nodes': len(winner.nodes),
           'connections': len(winner.connections)
       })
   
   fitnesses = [r['fitness'] for r in results]
   print(f"Mean fitness: {np.mean(fitnesses):.2f} ± {np.std(fitnesses):.2f}")
   print(f"Best fitness: {max(fitnesses):.2f}")
   print(f"Worst fitness: {min(fitnesses):.2f}")

Debugging Non-Reproducible Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're getting different results despite setting a seed:

1. **Check fitness function** - Does it use non-seeded randomness?

.. code-block:: python

   # Bad: unseeded numpy
   def eval_genomes(genomes, config):
       for gid, genome in genomes:
           genome.fitness = np.random.random()  # Not seeded!
   
   # Good: seeded numpy
   def eval_genomes(genomes, config):
       for gid, genome in genomes:
           genome.fitness = random.random()  # Uses NEAT's seed

2. **Check external dependencies** - Libraries with internal state?

3. **Check Python version** - Different versions may produce different sequences

4. **Check multiprocessing** - Using custom worker initialization?

Complete Examples
-----------------

Basic Reproducible XOR
^^^^^^^^^^^^^^^^^^^^^^

Here's a complete reproducible XOR example:

.. code-block:: python

   import os
   import neat
   
   # XOR inputs and expected outputs
   xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
   xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]
   
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           genome.fitness = 4.0
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           for xi, xo in zip(xor_inputs, xor_outputs):
               output = net.activate(xi)
               genome.fitness -= (output[0] - xo[0]) ** 2
   
   def run():
       # Load configuration
       local_dir = os.path.dirname(__file__)
       config_path = os.path.join(local_dir, 'config-xor')
       config = neat.Config(neat.DefaultGenome,
                           neat.DefaultReproduction,
                           neat.DefaultSpeciesSet,
                           neat.DefaultStagnation,
                           config_path)
       
       # Run with seed for reproducibility
       pop = neat.Population(config, seed=42)
       pop.add_reporter(neat.StdOutReporter(True))
       
       winner = pop.run(eval_genomes, 300)
       print(f"\\nWinner fitness: {winner.fitness:.6f}")
       return winner
   
   if __name__ == '__main__':
       # Run twice - should get identical results
       winner1 = run()
       winner2 = run()
       
       assert winner1.fitness == winner2.fitness
       print("\\n✓ Results are reproducible!")

Reproducible Parallel Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import os
   import multiprocessing
   import neat
   
   def eval_genome(genome, config):
       """Evaluate single genome - can use randomness."""
       import random  # Import in function for worker processes
       
       net = neat.nn.FeedForwardNetwork.create(genome, config)
       
       # Example: fitness depends on random inputs
       fitness = 0.0
       for _ in range(10):
           inputs = [random.random(), random.random()]
           outputs = net.activate(inputs)
           fitness += outputs[0]
       
       return fitness / 10.0
   
   def run():
       local_dir = os.path.dirname(__file__)
       config_path = os.path.join(local_dir, 'config-neat')
       config = neat.Config(neat.DefaultGenome,
                           neat.DefaultReproduction,
                           neat.DefaultSpeciesSet,
                           neat.DefaultStagnation,
                           config_path)
       
       # Use parallel evaluation with seed
       num_workers = multiprocessing.cpu_count()
       with neat.ParallelEvaluator(num_workers, eval_genome, seed=42) as evaluator:
           pop = neat.Population(config, seed=42)
           pop.add_reporter(neat.StdOutReporter(True))
           winner = pop.run(evaluator.evaluate, 100)
       
       return winner
   
   if __name__ == '__main__':
       winner1 = run()
       winner2 = run()
       
       print(f"Run 1 fitness: {winner1.fitness:.6f}")
       print(f"Run 2 fitness: {winner2.fitness:.6f}")
       assert winner1.fitness == winner2.fitness
       print("✓ Parallel evaluation is reproducible!")

Multi-Seed Statistical Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import os
   import numpy as np
   import neat
   
   def eval_genomes(genomes, config):
       # Your fitness function here
       pass
   
   def run_experiment(num_runs=10):
       """Run multiple experiments with different seeds."""
       config = neat.Config(...)  # Load your config
       
       results = []
       for i in range(num_runs):
           seed = 42 + i  # Different seed for each run
           
           pop = neat.Population(config, seed=seed)
           winner = pop.run(eval_genomes, 100)
           
           results.append({
               'seed': seed,
               'fitness': winner.fitness,
               'generation': pop.generation,
               'nodes': len(winner.nodes),
               'connections': len(winner.connections)
           })
           
           print(f"Run {i+1}/{num_runs}: "
                 f"fitness={winner.fitness:.2f}, "
                 f"gen={pop.generation}")
       
       return results
   
   def analyze_results(results):
       """Analyze multi-seed results."""
       fitnesses = [r['fitness'] for r in results]
       generations = [r['generation'] for r in results]
       
       print("\\n" + "="*60)
       print("RESULTS SUMMARY")
       print("="*60)
       print(f"Fitness:     {np.mean(fitnesses):.2f} ± {np.std(fitnesses):.2f}")
       print(f"  Best:      {max(fitnesses):.2f}")
       print(f"  Worst:     {min(fitnesses):.2f}")
       print(f"Generations: {np.mean(generations):.1f} ± {np.std(generations):.1f}")
       
       # Find best run
       best_idx = np.argmax(fitnesses)
       print(f"\\nBest run: seed={results[best_idx]['seed']}, "
             f"fitness={results[best_idx]['fitness']:.2f}")
   
   if __name__ == '__main__':
       results = run_experiment(num_runs=10)
       analyze_results(results)

See Also
--------

* :doc:`config_file` - Configuration file format and parameters
* :doc:`customization` - Customizing NEAT components
* :ref:`neat-overview-label` - Overview of the NEAT algorithm
* :doc:`module_summaries` - API reference for Population and ParallelEvaluator

Related Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``seed`` - Random seed for reproducibility (optional, default: None)
* ``pop_size`` - Population size affects random initialization
* ``reset_on_extinction`` - Affects behavior when population goes extinct

For complete configuration documentation, see :doc:`config_file`.
