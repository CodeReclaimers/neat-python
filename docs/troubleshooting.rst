Troubleshooting Guide
====================

This guide helps you diagnose and fix common problems when using NEAT-Python. Each section follows a **Symptoms → Causes → Solutions** structure.

Population Stuck at Low Fitness
--------------------------------

Symptoms
~~~~~~~~

- Fitness not improving after many generations (20-50+)
- Best fitness plateaued far from fitness threshold
- Average fitness not increasing
- Generation output shows no progress

Example output showing the problem:

.. code-block:: text

   Generation 0:   Best fitness: 2.1
   Generation 50:  Best fitness: 2.4
   Generation 100: Best fitness: 2.5  ← Still stuck
   Generation 150: Best fitness: 2.6  ← Barely improving

Causes
~~~~~~

**1. Fitness function not working correctly**

Most common issue! Check:

- Are you setting ``genome.fitness`` for every genome?
- Are fitness values reasonable (not all the same)?
- Does better performance = higher fitness?

**2. Insufficient genetic diversity**

- Population too small
- Compatibility threshold too high (too few species)
- Population converged to local optimum

**3. Network structure cannot represent solution**

- Wrong activation function for the problem
- ``feed_forward = True`` when problem needs memory
- Initial connections wrong (``none`` when should be ``full``)

**4. Problem is genuinely hard**

- Fitness threshold set unrealistically high
- Problem requires many generations to solve

Solutions
~~~~~~~~~

**Step 1: Debug your fitness function**

.. code-block:: python

   def eval_genomes(genomes, config):
       fitness_values = []
       
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           genome.fitness = calculate_fitness(net)
           fitness_values.append(genome.fitness)
           
           # Debug: Print some fitness values
           if len(fitness_values) <= 5:
               print(f"Genome {genome_id}: fitness = {genome.fitness}")
       
       # Check fitness distribution
       print(f"Fitness range: {min(fitness_values):.2f} to {max(fitness_values):.2f}")
       print(f"Fitness average: {sum(fitness_values)/len(fitness_values):.2f}")

**What to look for:**
- All fitness values are ``None`` → You're not setting fitness
- All fitness values identical → Fitness function not differentiating
- Fitness decreases with better performance → Sign is backwards

**Step 2: Increase diversity**

.. code-block:: ini

   [NEAT]
   # Increase population size
   pop_size = 300  # Was: 150
   
   [DefaultSpeciesSet]
   # Decrease compatibility threshold (more species = more diversity)
   compatibility_threshold = 2.5  # Was: 3.0

**Step 3: Check activation functions**

.. code-block:: ini

   [DefaultGenome]
   # If problem needs [-1, 1] outputs, use tanh not sigmoid
   activation_default = tanh
   
   # Allow evolution to explore different activations
   activation_mutate_rate = 0.1
   activation_options = tanh sigmoid relu

**Step 4: Check network type**

.. code-block:: ini

   [DefaultGenome]
   # For sequential/temporal problems, allow recurrence
   feed_forward = False  # Was: True
   
   # Start with some hidden nodes
   num_hidden = 2  # Was: 0

**Step 5: Visualize evolution progress**

.. code-block:: python

   from neat import StatisticsReporter
   
   stats = StatisticsReporter()
   p.add_reporter(stats)
   winner = p.run(eval_genomes, 300)
   
   # Plot fitness over time
   # (Copy visualize.py from examples/xor/)
   import visualize
   visualize.plot_stats(stats, ylog=False, view=True)
   visualize.plot_species(stats, view=True)

**See also:** :doc:`cookbook` section on "Debug Population Not Evolving"

All Species Went Extinct
-------------------------

Symptoms
~~~~~~~~

Error message during evolution:

.. code-block:: text

   RuntimeError: All species have gone extinct

Or in generation output:

.. code-block:: text

   Generation 15: Population of 0 members in 0 species
   Total extinctions: 3

Causes
~~~~~~

**1. All fitness values ≤ 0**

If every genome has fitness ≤ 0, NEAT can't select parents for reproduction.

**2. Population too small**

With very small populations, random chance can eliminate all species.

**3. Compatibility threshold too low**

Creates too many tiny species that can't survive.

**4. Stagnation threshold too aggressive**

Species removed before they can improve.

Solutions
~~~~~~~~~

**Solution 1: Ensure positive fitness**

.. code-block:: python

   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           
           # Calculate fitness (might be negative)
           raw_fitness = calculate_fitness(net)
           
           # Ensure it's positive
           genome.fitness = max(0.001, raw_fitness)
           
           # Or shift into positive range
           genome.fitness = raw_fitness + 100.0

**Solution 2: Increase population size**

.. code-block:: ini

   [NEAT]
   pop_size = 150  # Minimum recommended
   
   # For complex problems
   pop_size = 300

**Solution 3: Adjust compatibility threshold**

.. code-block:: ini

   [DefaultSpeciesSet]
   # Increase to reduce number of species
   compatibility_threshold = 4.0  # Was: 2.0

**Solution 4: Adjust stagnation settings**

.. code-block:: ini

   [DefaultStagnation]
   # Allow more generations before removing species
   max_stagnation = 30  # Was: 15
   
   # Protect more top species
   species_elitism = 3  # Was: 2

**Solution 5: Enable extinction recovery**

.. code-block:: ini

   [NEAT]
   # Create new random population if all species extinct
   reset_on_extinction = True

**Prevention:** Monitor species count during evolution:

.. code-block:: python

   class SpeciesMonitor(neat.reporting.BaseReporter):
       def post_evaluate(self, config, population, species, best_genome):
           print(f"  Species count: {len(species.species)}")
   
   p.add_reporter(SpeciesMonitor())

Network Complexity Exploding
-----------------------------

Symptoms
~~~~~~~~

- Networks have hundreds of nodes after just a few generations
- Evolution becomes very slow
- Networks are too complex to understand/interpret
- Fitness improves but networks are unnecessarily large

Example:

.. code-block:: text

   Generation 10:  Best fitness: 3.2 - size: (5, 12)    ← Reasonable
   Generation 20:  Best fitness: 3.5 - size: (23, 67)   ← Growing fast
   Generation 30:  Best fitness: 3.7 - size: (89, 234)  ← Exploding!

Causes
~~~~~~

**1. High mutation rates**

- ``conn_add_prob`` and ``node_add_prob`` too high
- Deletion rates too low

**2. No selection pressure for simplicity**

- Fitness function only rewards task performance
- No penalty for network size

**3. Initial configuration**

- Starting with too many hidden nodes
- ``initial_connection = full`` on large networks

Solutions
~~~~~~~~~

**Solution 1: Reduce mutation rates**

.. code-block:: ini

   [DefaultGenome]
   # Reduce addition probabilities
   conn_add_prob = 0.3      # Default: 0.5
   node_add_prob = 0.1      # Default: 0.2
   
   # Increase deletion probabilities
   conn_delete_prob = 0.7   # Default: 0.5
   node_delete_prob = 0.5   # Default: 0.2

**Solution 2: Add complexity penalty to fitness**

.. code-block:: python

   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           
           # Task performance (e.g., 0-4 for XOR)
           task_fitness = evaluate_task(net)
           
           # Count connections and non-input nodes
           num_connections = len(genome.connections)
           num_nodes = len([n for n in genome.nodes.values()
                           if n.key not in config.genome_config.input_keys])
           
           # Penalty for complexity
           complexity_penalty = 0.01 * (num_connections + num_nodes)
           
           # Final fitness
           genome.fitness = task_fitness - complexity_penalty

**Adjust penalty weight:**
- Stronger penalty (0.1): Strongly favor small networks
- Weaker penalty (0.001): Slightly favor small networks
- Adaptive penalty: Increase penalty as task_fitness improves

**Solution 3: Limit structural mutations**

.. code-block:: ini

   [DefaultGenome]
   # Allow only one structural mutation per genome per generation
   single_structural_mutation = true

**Solution 4: Start simpler**

.. code-block:: ini

   [DefaultGenome]
   num_hidden = 0              # Start with no hidden nodes
   initial_connection = full   # Fully connect inputs to outputs

**See also:** :doc:`cookbook` section on "Control Network Complexity"

Checkpoint Restore Errors
--------------------------

Symptoms
~~~~~~~~

Errors when loading checkpoints:

.. code-block:: text

   AttributeError: 'DefaultGenome' object has no attribute 'innovation_tracker'
   
   FileNotFoundError: neat-checkpoint-50

   pickle.UnpicklingError: invalid load key

Causes
~~~~~~

**1. Version incompatibility**

- Trying to load v0.x checkpoint in v1.0+
- Different NEAT-Python versions
- Innovation tracking incompatibility

**2. Corrupted checkpoint file**

- File partially written (evolution interrupted)
- Disk errors
- Wrong file format

**3. Missing dependencies**

- Config file not found
- Custom classes not importable
- Missing modules

Solutions
~~~~~~~~~

**Solution 1: Check version compatibility**

.. code-block:: python

   import neat
   print(f"NEAT-Python version: {neat.__version__}")
   
   # v1.0+ checkpoints NOT compatible with v0.x
   # v0.x checkpoints NOT compatible with v1.0+

.. warning::
   **Breaking change in v1.0.0:** Checkpoints from v0.93 and earlier are **not compatible** with v1.0.0+ due to innovation number tracking.
   
   **Solution:** Use v0.93 to finish old runs, or start fresh with v1.0.0.

**Solution 2: Verify checkpoint file exists**

.. code-block:: python

   import os
   
   checkpoint_file = 'neat-checkpoint-50'
   if os.path.exists(checkpoint_file):
       print(f"Found checkpoint: {checkpoint_file}")
       print(f"Size: {os.path.getsize(checkpoint_file)} bytes")
   else:
       print(f"Checkpoint not found: {checkpoint_file}")
       print(f"Available checkpoints: {os.listdir('.')}")

**Solution 3: Use absolute paths**

.. code-block:: python

   import os
   
   # ❌ Relative path may fail
   p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-50')
   
   # ✅ Absolute path
   checkpoint_dir = os.path.dirname(__file__)
   checkpoint_path = os.path.join(checkpoint_dir, 'neat-checkpoint-50')
   p = neat.Checkpointer.restore_checkpoint(checkpoint_path)

**Solution 4: Handle corrupted files**

.. code-block:: python

   import pickle
   
   # Try loading checkpoint
   try:
       p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-50')
       print("Checkpoint loaded successfully")
   except (pickle.UnpicklingError, EOFError, AttributeError) as e:
       print(f"Checkpoint corrupted: {e}")
       print("Try loading an earlier checkpoint")
       p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-45')

**Best practices:**

.. code-block:: python

   # Save checkpoints frequently
   checkpointer = neat.Checkpointer(generation_interval=5)  # Every 5 gen
   p.add_reporter(checkpointer)
   
   # Keep multiple checkpoints (last 10)
   # Manual cleanup if needed
   import glob
   checkpoints = sorted(glob.glob('neat-checkpoint-*'))
   if len(checkpoints) > 10:
       for old_checkpoint in checkpoints[:-10]:
           os.remove(old_checkpoint)

ModuleNotFoundError for Examples
---------------------------------

Symptoms
~~~~~~~~

When running examples:

.. code-block:: text

   ModuleNotFoundError: No module named 'visualize'
   ModuleNotFoundError: No module named 'gym'
   ModuleNotFoundError: No module named 'numpy'

Causes
~~~~~~

**1. Missing dependencies**

Examples have additional dependencies beyond core NEAT-Python.

**2. Wrong directory**

Running examples from wrong location.

**3. Environment not activated**

Using system Python instead of project environment.

Solutions
~~~~~~~~~

**Solution 1: Install example dependencies**

.. code-block:: bash

   # Install all example dependencies
   pip install -r examples/requirements.txt

**Solution 2: Use conda environment**

.. code-block:: bash

   # Create environment from file
   conda env create -f examples/environment.yml
   
   # Activate environment
   conda activate neat-python-examples

**Solution 3: Install specific dependencies**

.. code-block:: bash

   # For visualization (XOR, pole balancing)
   pip install graphviz matplotlib
   
   # For OpenAI Gym examples
   pip install gym numpy
   
   # For all examples
   pip install graphviz matplotlib gym numpy

**Solution 4: Copy visualize.py to your project**

If you just need the visualization functions:

.. code-block:: bash

   # Copy from XOR example
   cp examples/xor/visualize.py your_project/

Then in your code:

.. code-block:: python

   import visualize  # Now works in your project
   visualize.plot_stats(stats, view=True)

Fitness Function Errors
------------------------

Symptoms
~~~~~~~~

**All fitness values are None:**

.. code-block:: text

   Best fitness: None - size: (0, 2)

**Exception during fitness evaluation:**

.. code-block:: text

   TypeError: unsupported operand type(s) for -: 'NoneType' and 'float'
   IndexError: list index out of range

**Fitness values unexpected:**

- All zeros
- All the same value
- Negative when should be positive

Causes
~~~~~~

**1. Not setting genome.fitness**

Most common mistake!

**2. Returning instead of setting**

Confusion with ``ParallelEvaluator`` signature.

**3. Network activation errors**

Wrong input size, wrong output indexing.

**4. Fitness calculation errors**

Division by zero, None values in calculation.

Solutions
~~~~~~~~~

**Solution 1: Always set genome.fitness**

.. code-block:: python

   # ✅ Correct
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           genome.fitness = calculate_fitness(net)  # Set it!
   
   # ❌ Wrong - not setting fitness
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           result = calculate_fitness(net)  # Calculated but not assigned!

**Solution 2: Check network activation**

.. code-block:: python

   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           try:
               net = neat.nn.FeedForwardNetwork.create(genome, config)
               
               # Verify input size matches config
               num_inputs = len(config.genome_config.input_keys)
               inputs = [1.0] * num_inputs  # Create correct size input
               
               output = net.activate(inputs)
               
               # Verify output size
               num_outputs = len(config.genome_config.output_keys)
               assert len(output) == num_outputs
               
               genome.fitness = calculate_fitness(output)
               
           except Exception as e:
               print(f"Error evaluating genome {genome_id}: {e}")
               genome.fitness = 0.0  # Assign default fitness on error

**Solution 3: Validate fitness values**

.. code-block:: python

   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           
           raw_fitness = calculate_fitness(net)
           
           # Validate fitness is valid number
           if raw_fitness is None or math.isnan(raw_fitness):
               genome.fitness = 0.0
               print(f"Warning: Invalid fitness for genome {genome_id}")
           else:
               genome.fitness = max(0.0, raw_fitness)  # Ensure non-negative

**Solution 4: Debug with print statements**

.. code-block:: python

   def eval_genomes(genomes, config):
       for i, (genome_id, genome) in enumerate(genomes):
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           genome.fitness = calculate_fitness(net)
           
           # Debug first few genomes
           if i < 3:
               print(f"Genome {genome_id}:")
               print(f"  Nodes: {len(genome.nodes)}")
               print(f"  Connections: {len(genome.connections)}")
               print(f"  Fitness: {genome.fitness}")

**Solution 5: Handle ParallelEvaluator correctly**

.. code-block:: python

   # For ParallelEvaluator: RETURN fitness, don't set it
   def eval_genome(genome, config):  # Note: single genome
       net = neat.nn.FeedForwardNetwork.create(genome, config)
       fitness = calculate_fitness(net)
       return fitness  # Return, don't set genome.fitness!
   
   # Use with ParallelEvaluator
   with neat.ParallelEvaluator(4, eval_genome) as evaluator:
       winner = p.run(evaluator.evaluate, 300)

**See also:** :doc:`xor_example` for working fitness function example

Getting More Help
-----------------

If you're still stuck after trying these solutions:

**1. Check the FAQ**

:doc:`faq` - Common questions about NEAT-Python

**2. Review the examples**

Look at working examples in the ``examples/`` directory:
- ``xor/`` - Simple complete example
- ``single-pole-balancing/`` - Control problem
- ``openai-lander/`` - Gym integration

**3. Enable verbose output**

.. code-block:: python

   # Add stdout reporter for detailed progress
   p.add_reporter(neat.StdOutReporter(True))  # True = verbose
   
   # Add statistics for analysis
   stats = neat.StatisticsReporter()
   p.add_reporter(stats)

**4. Check GitHub Issues**

Search existing issues: https://github.com/CodeReclaimers/neat-python/issues

**5. Create a minimal reproducible example**

When asking for help, provide:
- NEAT-Python version
- Minimal code that shows the problem
- Config file
- Error message or unexpected behavior
- What you've already tried

**See also:**
- :doc:`cookbook` - Practical recipes
- :doc:`faq` - Frequently asked questions
- :doc:`config_file` - Configuration reference
