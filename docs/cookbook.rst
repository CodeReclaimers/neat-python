Cookbook: Common Patterns
=========================

This cookbook provides practical solutions to common problems and patterns when using NEAT-Python. Each recipe includes working code you can copy and adapt for your own projects.

How to: Set Specific Output Activation Functions
-------------------------------------------------

**Problem:** You need network outputs in a specific range (e.g., [-1, 1] for control problems).

**Solution:** Configure the activation function in your config file:

.. code-block:: ini

   [DefaultGenome]
   # For outputs in range [-1, 1]
   activation_default = tanh
   activation_mutate_rate = 0.0  # Don't change activation
   activation_options = tanh     # Only allow tanh

**Alternative:** If you need to mix activation functions:

.. code-block:: ini

   # Allow multiple options, set mutation rate > 0
   activation_default = tanh
   activation_mutate_rate = 0.2
   activation_options = tanh sigmoid relu

**Post-processing approach:**

.. code-block:: python

   import math
   
   # Get network output
   output = net.activate(inputs)
   
   # Transform to desired range
   output_tanh = [math.tanh(x) for x in output]  # [-1, 1]
   output_sigmoid = [1.0 / (1.0 + math.exp(-x)) for x in output]  # [0, 1]

.. warning::
   Make sure your fitness function expects the same range as your activation function outputs!

**See also:** :doc:`activation` for all available activation functions.

How to: Use Parallel Evaluation
--------------------------------

**Problem:** Fitness evaluation is slow and you want to use multiple CPU cores.

**Solution:** Use ``ParallelEvaluator`` with a context manager for automatic cleanup:

.. code-block:: python

   import neat
   import multiprocessing
   
   def eval_genome(genome, config):
       """
       Evaluate a single genome.
       IMPORTANT: Must return fitness value (not set genome.fitness).
       """
       net = neat.nn.FeedForwardNetwork.create(genome, config)
       
       # Your fitness evaluation here
       fitness = 0.0
       for test_case in test_cases:
           output = net.activate(test_case.inputs)
           fitness += calculate_score(output, test_case.expected)
       
       return fitness  # Return, don't set genome.fitness
   
   # Use context manager for proper cleanup
   config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-file')
   
   p = neat.Population(config)
   p.add_reporter(neat.StdOutReporter(True))
   
   with neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome) as evaluator:
       winner = p.run(evaluator.evaluate, 300)
   # Pool automatically cleaned up here

**Common mistakes:**

.. warning::
   ❌ **Wrong:** Setting ``genome.fitness`` in eval_genome
   
   .. code-block:: python
   
      def eval_genome(genome, config):
          genome.fitness = 10.0  # Don't do this!
   
   ✅ **Right:** Returning fitness value
   
   .. code-block:: python
   
      def eval_genome(genome, config):
          return 10.0  # Return the value

.. warning::
   ❌ **Wrong:** Forgetting to use context manager (memory leak)
   
   .. code-block:: python
   
      evaluator = neat.ParallelEvaluator(4, eval_genome)
      winner = p.run(evaluator.evaluate, 300)
      # Pool never cleaned up!
   
   ✅ **Right:** Using ``with`` statement
   
   .. code-block:: python
   
      with neat.ParallelEvaluator(4, eval_genome) as evaluator:
          winner = p.run(evaluator.evaluate, 300)

**Performance tip:** Start with ``multiprocessing.cpu_count()`` and adjust based on your workload.

**See also:** :doc:`module_summaries` for ``ParallelEvaluator`` API details.

How to: Save and Restore Checkpoints
-------------------------------------

**Problem:** You want to save evolution progress and resume later.

**Solution:** Use the ``Checkpointer`` reporter:

.. code-block:: python

   import neat
   
   config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-file')
   
   p = neat.Population(config)
   p.add_reporter(neat.StdOutReporter(True))
   
   # Save checkpoint every 5 generations
   checkpointer = neat.Checkpointer(generation_interval=5,
                                     time_interval_seconds=None,
                                     filename_prefix='neat-checkpoint-')
   p.add_reporter(checkpointer)
   
   # Run evolution
   winner = p.run(eval_genomes, 100)

This creates files: ``neat-checkpoint-0``, ``neat-checkpoint-5``, ``neat-checkpoint-10``, etc.

**Restoring from checkpoint:**

.. code-block:: python

   import neat
   
   # Restore from generation 50
   p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-50')
   
   # Continue evolution for 50 more generations
   winner = p.run(eval_genomes, 50)

**Time-based checkpointing:**

.. code-block:: python

   # Save every 10 minutes instead of every N generations
   checkpointer = neat.Checkpointer(generation_interval=None,
                                     time_interval_seconds=600)

.. note::
   **Checkpoint compatibility:** Checkpoints from v1.0+ are **not compatible** with v0.x due to innovation number tracking. See :doc:`migration` for details.

**See also:** :py:class:`neat.Checkpointer <checkpoint.Checkpointer>` API documentation.

How to: Debug "Population Not Evolving"
----------------------------------------

**Problem:** Fitness isn't improving over many generations.

**Diagnostic steps:**

**1. Check fitness function is working**

.. code-block:: python

   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           genome.fitness = calculate_fitness(net)
           
           # Debug: Print fitness values
           if genome_id % 10 == 0:  # Print every 10th genome
               print(f"Genome {genome_id}: fitness = {genome.fitness}")

**Check for these issues:**

- ✅ All genomes have valid fitness (not ``None``)
- ✅ Fitness values vary between genomes
- ✅ Better performance = higher fitness

**2. Check for sufficient diversity**

.. code-block:: ini

   [NEAT]
   pop_size = 150  # Try increasing (e.g., 300)
   
   [DefaultSpeciesSet]
   compatibility_threshold = 3.0  # Try decreasing (e.g., 2.0)

Lower compatibility threshold = more species = more diversity.

**3. Check activation functions**

.. code-block:: ini

   [DefaultGenome]
   # If problem needs outputs in [-1, 1], use tanh not sigmoid
   activation_default = tanh  # Not sigmoid!
   
   # Allow evolution to try different activations
   activation_mutate_rate = 0.1
   activation_options = tanh sigmoid relu

**4. Add diagnostic reporters**

.. code-block:: python

   stats = neat.StatisticsReporter()
   p.add_reporter(stats)
   p.add_reporter(neat.StdOutReporter(True))
   
   winner = p.run(eval_genomes, 300)
   
   # Visualize fitness over time
   import visualize  # From examples/xor/visualize.py
   visualize.plot_stats(stats, ylog=False, view=True)

**See also:** :doc:`troubleshooting` for more diagnostic techniques.

How to: Interpret Fitness Trends
---------------------------------

**Problem:** You want to understand if evolution is progressing well.

**Good fitness curve (steady improvement):**

.. code-block:: text

   Generation 0:   Best fitness = 2.1   Avg = 1.5
   Generation 10:  Best fitness = 3.2   Avg = 2.3
   Generation 20:  Best fitness = 3.7   Avg = 2.9
   Generation 30:  Best fitness = 3.9   Avg = 3.4  ← Converging
   Generation 40:  Best fitness = 3.95  Avg = 3.7  ← Near threshold

**What this means:** 
- ✅ Best fitness steadily increasing
- ✅ Average fitness following behind
- ✅ Gap between best and average narrowing (population converging)

**Bad fitness curve (stuck):**

.. code-block:: text

   Generation 0:   Best fitness = 2.1   Avg = 1.5
   Generation 10:  Best fitness = 2.3   Avg = 1.6
   Generation 20:  Best fitness = 2.4   Avg = 1.7
   Generation 30:  Best fitness = 2.3   Avg = 1.6  ← No improvement
   Generation 40:  Best fitness = 2.5   Avg = 1.8  ← Stuck!

**Diagnosis:** Population stuck in local optimum.

**Solutions:**
1. Increase population size or diversity
2. Check if network structure can represent solution
3. Verify fitness function rewards improvement

**Volatile fitness curve (unstable):**

.. code-block:: text

   Generation 0:   Best fitness = 2.1   Avg = 1.5
   Generation 10:  Best fitness = 3.8   Avg = 1.9
   Generation 20:  Best fitness = 2.9   Avg = 1.7  ← Dropped!
   Generation 30:  Best fitness = 3.6   Avg = 2.1
   Generation 40:  Best fitness = 3.0   Avg = 1.8  ← Volatile

**Diagnosis:** Fitness function is noisy or non-deterministic.

**Solutions:**

.. code-block:: python

   # 1. Average over multiple trials
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           
           # Run multiple trials and average
           fitness_values = []
           for trial in range(3):  # 3 trials
               fitness_values.append(run_trial(net))
           
           genome.fitness = sum(fitness_values) / len(fitness_values)
   
   # 2. Use elitism to preserve best genomes
   # In config file:
   # [DefaultReproduction]
   # elitism = 5  # Always keep 5 best genomes

**When to adjust parameters:**

- **Too slow:** Increase ``pop_size`` or decrease ``compatibility_threshold``
- **Too fast (premature convergence):** Decrease ``pop_size`` or increase ``compatibility_threshold``
- **Complexity exploding:** See next section

How to: Control Network Complexity
-----------------------------------

**Problem:** Networks are growing too large (hundreds of nodes/connections).

**Solution 1: Adjust mutation rates**

.. code-block:: ini

   [DefaultGenome]
   # Reduce addition rates
   conn_add_prob = 0.3      # Default: 0.5
   node_add_prob = 0.1      # Default: 0.2
   
   # Increase deletion rates
   conn_delete_prob = 0.7   # Default: 0.5
   node_delete_prob = 0.4   # Default: 0.2

**Solution 2: Add complexity penalty to fitness**

.. code-block:: python

   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           
           # Calculate task performance
           task_fitness = evaluate_task(net)
           
           # Penalize complexity
           num_connections = len(genome.connections)
           num_nodes = len([n for n in genome.nodes.values() 
                           if n.key not in config.genome_config.input_keys])
           
           complexity_penalty = 0.01 * (num_connections + num_nodes)
           
           genome.fitness = task_fitness - complexity_penalty

**Adjust penalty weight (0.01) based on your needs:**
- Larger penalty (0.1) = strong preference for small networks
- Smaller penalty (0.001) = weak preference, allow larger networks

**Solution 3: Start with hidden nodes**

.. code-block:: ini

   [DefaultGenome]
   num_hidden = 2  # Start with 2 hidden nodes
   initial_connection = full  # Fully connected

This can help find solutions faster without excessive growth.

**See also:** :doc:`config_file` for all mutation parameters.

How to: Handle Different Output Ranges
---------------------------------------

**Problem:** Your problem requires specific output ranges.

**Common ranges and solutions:**

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Output Range
     - Activation Function
     - Use Case
   * - [0, 1]
     - sigmoid
     - Probabilities, binary classification
   * - [-1, 1]
     - tanh
     - Control signals, normalized values
   * - [0, ∞)
     - relu, softplus
     - Non-negative values, quantities
   * - Any range
     - identity + scaling
     - Custom ranges

**Example: Control problem needing [-1, 1]**

.. code-block:: ini

   [DefaultGenome]
   activation_default = tanh
   activation_mutate_rate = 0.0
   activation_options = tanh

**Example: Multi-output with different ranges**

.. code-block:: python

   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           raw_output = net.activate(inputs)
           
           # Assume 3 outputs: probability, control, quantity
           probability = sigmoid(raw_output[0])  # [0, 1]
           control = tanh(raw_output[1])          # [-1, 1]
           quantity = relu(raw_output[2])         # [0, ∞)
           
           genome.fitness = evaluate(probability, control, quantity)

**Example: Custom range scaling**

.. code-block:: python

   # Want outputs in [5, 15]
   raw_output = net.activate(inputs)[0]  # From sigmoid: [0, 1]
   scaled_output = 5.0 + raw_output * 10.0  # Scale to [5, 15]

**See also:** :doc:`activation` for all available activation functions and their ranges.

How to: Configure for Different Problem Types
----------------------------------------------

**Feedforward vs. Recurrent:**

**Use feedforward when:**
- Problem has no temporal dependencies
- Each input → output is independent
- Examples: XOR, classification, function approximation

.. code-block:: ini

   [DefaultGenome]
   feed_forward = True
   initial_connection = full

**Use recurrent when:**
- Problem requires memory of past inputs
- Time-series or sequential data
- Examples: control, game playing, sequence prediction

.. code-block:: ini

   [DefaultGenome]
   feed_forward = False
   initial_connection = partial  # Or full

**Simple vs. Complex problems:**

**Simple problems (like XOR):**

.. code-block:: ini

   [NEAT]
   pop_size = 50                # Smaller population
   fitness_threshold = 3.9      # Clear target
   
   [DefaultGenome]
   num_hidden = 0               # Start minimal
   conn_add_prob = 0.5
   node_add_prob = 0.2

**Complex problems (like game playing):**

.. code-block:: ini

   [NEAT]
   pop_size = 300               # Larger population
   no_fitness_termination = False  # May not reach threshold
   
   [DefaultGenome]
   num_hidden = 2               # Start with some complexity
   conn_add_prob = 0.7          # Allow faster growth
   node_add_prob = 0.3

**Fast exploration vs. thorough search:**

**Fast exploration (quick results):**

.. code-block:: ini

   [NEAT]
   pop_size = 50
   
   [DefaultGenome]
   conn_add_prob = 0.7          # Aggressive complexification
   node_add_prob = 0.3
   
   [DefaultSpeciesSet]
   compatibility_threshold = 4.0  # Fewer species

**Thorough search (best solution):**

.. code-block:: ini

   [NEAT]
   pop_size = 300
   
   [DefaultGenome]
   conn_add_prob = 0.4          # Slower complexification
   node_add_prob = 0.15
   
   [DefaultSpeciesSet]
   compatibility_threshold = 2.5  # More species

**See also:** 
- :doc:`config_essentials` for parameter explanations
- :doc:`neat_overview` for algorithm understanding
- :doc:`xor_example` for a complete simple example

Common Gotchas
--------------

**1. Forgetting to set genome.fitness**

This is the #1 mistake! NEAT can't evolve without fitness values.

.. code-block:: python

   # ❌ Wrong
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           result = net.activate([1, 0])
           # Forgot to set fitness!
   
   # ✅ Right
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           result = net.activate([1, 0])
           genome.fitness = calculate_fitness(result)

**2. Negative fitness values causing extinction**

If all genomes have fitness ≤ 0, species can go extinct.

.. code-block:: python

   # ✅ Ensure fitness is positive
   genome.fitness = max(0.001, calculated_fitness)

**3. Config file in wrong location**

.. code-block:: python

   # ❌ Relative path may fail
   config = neat.Config(..., 'config-file')
   
   # ✅ Use absolute path
   import os
   local_dir = os.path.dirname(__file__)
   config_path = os.path.join(local_dir, 'config-file')
   config = neat.Config(..., config_path)

**4. Not using context managers with ParallelEvaluator**

Always use ``with`` statement to ensure proper cleanup:

.. code-block:: python

   # ✅ Right
   with neat.ParallelEvaluator(4, eval_genome) as evaluator:
       winner = p.run(evaluator.evaluate, 300)

**5. Mixing up node keys**

- Input nodes: negative keys (-1, -2, ...)
- Output nodes: zero and positive keys (0, 1, ...)
- Hidden nodes: positive keys assigned during evolution

Next Steps
----------

- :doc:`troubleshooting` - Diagnostic help for common problems
- :doc:`faq` - Frequently asked questions
- :doc:`config_file` - Complete configuration reference
- :doc:`customization` - Advanced customization options
