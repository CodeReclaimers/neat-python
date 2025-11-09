
XOR Example: Detailed Walkthrough
==================================

.. default-role:: any

This is a complete walkthrough of the XOR example (``examples/xor/evolve-feedforward.py``), which evolves a neural network to compute the XOR (exclusive-or) logic function. This is often considered the "Hello, World!" of evolutionary neural networks.

**Expected completion time:** Usually converges in 50-150 generations, taking 1-3 seconds on modern hardware.

The XOR Problem
---------------

XOR (exclusive-or) is a classic test problem for neural networks:

=======  =======  ======
Input 1  Input 2  Output
=======  =======  ======
0        0        0
0        1        1
1        0        1
1        1        0
=======  =======  ======

.. index:: ! fitness function

Fitness function
----------------

The key thing you need to figure out for a given problem is how to measure the fitness of the :term:`genomes <genome>` that are produced
by NEAT.  Fitness is expected to be a Python `float` value.  If genome A solves your problem more successfully than genome B,
then the fitness value of A should be greater than the value of B.  The absolute magnitude and signs of these fitnesses
are not important, only their relative values.

In this example, we create a :term:`feed-forward` neural network based on the genome, and then for each case in the
table above, we provide that network with the inputs, and compute the network's output.  The error for each genome
is :math:`1 - \sum_i (e_i - a_i)^2` between the expected (:math:`e_i`) and actual (:math:`a_i`) outputs, so that if the
network produces exactly the expected output, its fitness is 1, otherwise it is a value less than 1, with the fitness
value decreasing the more incorrect the network responses are.

This fitness computation is implemented in the ``eval_genomes`` function.  This function takes two arguments: a list
of genomes (the current population) and the active configuration.  neat-python expects the fitness function to calculate
a fitness for each genome and assign this value to the genome's ``fitness`` member.

.. note::
   **What's Happening in the Fitness Function?**
   
   1. **Create Network**: Convert the genome (genetic encoding) into an actual neural network
   2. **Initialize Fitness**: Start with perfect fitness (4.0) since there are 4 test cases
   3. **Test All Cases**: Feed each XOR input to the network and get the output
   4. **Calculate Error**: Subtract the squared error from fitness
   5. **Final Fitness**: Higher fitness = better network. Fitness of 4.0 would be perfect.
   
   NEAT uses these fitness values to determine which genomes should reproduce. Better genomes are more likely to have offspring.

Sample Output
-------------

When you run the XOR example, you'll see generation-by-generation progress like this:

.. code-block:: text

   ****** Running generation 0 ******

   Population's average fitness: 2.21888 stdev: 0.34917
   Best fitness: 2.98110 - size: (1, 2) - species 1 - id 143
   Average adjusted fitness: 0.561
   Mean genetic distance 1.739, standard deviation 0.497
   Population of 150 members in 2 species (after reproduction):
      ID   age  size   fitness   adj fit  stag
     ====  ===  ====  =========  =======  ====
        1    0   123      2.981    0.599     0
        2    0    27      2.919    0.522     0
   Total extinctions: 0
   Generation time: 0.003 sec

**What this output means:**

- **Best fitness: 2.98110** - The best genome scored 2.98 out of 4.0 possible (pretty good for generation 0!)
- **size: (1, 2)** - This genome has 1 hidden node and 2 connections
- **species 1 - id 143** - This is genome #143 in species #1
- **2 species** - Population has split into 2 groups of similar genomes
- **Generation time: 0.003 sec** - Very fast! Fitness evaluation is simple for XOR

After some generations (typically 50-150), you'll see:

.. code-block:: text

   ****** Running generation 73 ******

   Best individual in generation 73 meets fitness threshold - complexity: (0, 2)

   Best genome:
   Key: 2891
   Fitness: 3.9623
   Nodes:
       0 DefaultNodeGene(key=0, bias=-3.127, response=1.0, activation=sigmoid, aggregation=sum)
   Connections:
       DefaultConnectionGene(key=(-1, 0), weight=4.723, enabled=True, innovation=1)
       DefaultConnectionGene(key=(-2, 0), weight=-4.856, enabled=True, innovation=2)

**Success!** The network reached the fitness threshold (3.9). Notice:

- **complexity: (0, 2)** - No hidden nodes needed! Direct connections from inputs to output were sufficient
- **2 connections** - From input -1 (first input) and input -2 (second input) to output 0
- **High weights** (±4.7) - Strong connections that saturate the sigmoid function
- **Negative bias** (-3.127) - Shifts the sigmoid to implement XOR logic

Common Mistakes
---------------

**1. Forgetting to Set genome.fitness**

.. code-block:: python

   # WRONG - fitness never assigned
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           output = net.activate([1.0, 0.0])
           # Oops! Forgot to set genome.fitness
   
   # RIGHT - fitness properly assigned
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           genome.fitness = calculate_fitness(net)  # ✓

**Symptom:** All genomes have None as fitness, evolution doesn't work

**2. Configuration File Path Issues**

.. code-block:: python

   # WRONG - relative path may not work depending on where you run from
   config = neat.Config(..., 'config-feedforward')
   
   # RIGHT - use absolute path
   import os
   local_dir = os.path.dirname(__file__)
   config_path = os.path.join(local_dir, 'config-feedforward')
   config = neat.Config(..., config_path)

**Symptom:** ``FileNotFoundError: config-feedforward``

**3. Wrong Activation Function Range**

.. code-block:: ini

   # If your fitness function expects outputs in [-1, 1]
   # but you're using sigmoid (outputs in [0, 1])
   activation_default = sigmoid  # Wrong for [-1, 1] range
   
   # Use tanh instead
   activation_default = tanh  # Correct for [-1, 1] range

**Symptom:** Networks never reach good fitness

**4. Not Returning Network Output**

.. code-block:: python

   # WRONG - forgot to use network output
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           net.activate(xor_inputs[0])  # Output discarded!
           genome.fitness = 1.0  # Fixed value - no feedback
   
   # RIGHT - use output to calculate fitness
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           output = net.activate(xor_inputs[0])  # ✓
           genome.fitness = 4.0 - (output[0] - expected) ** 2  # ✓

**Symptom:** All genomes get same fitness, no evolution

Running NEAT
------------

Once you have implemented a fitness function, you mostly just need some additional boilerplate code that carries out
the following steps:

* Create a :py:class:`neat.config.Config <config.Config>` object from the configuration file (described in the :doc:`config_file`).

.. note::
   **What's Happening: Configuration Loading**
   
   The config file controls everything about evolution:
   
   - **Population size**: How many genomes per generation
   - **Network structure**: Number of inputs/outputs, allowed activation functions
   - **Mutation rates**: How fast networks complexify
   - **Speciation**: How genetic diversity is protected
   - **Termination**: When to stop evolving
   
   See :doc:`config_essentials` for a beginner-friendly guide to configuration.

* Create a :py:class:`neat.population.Population <population.Population>` object using the ``Config`` object created above.

.. note::
   **What's Happening: Population Initialization**
   
   NEAT creates the initial population of random genomes (neural networks). Each genome starts simple - typically just inputs connected directly to outputs with random weights. Complexity evolves over time as needed.

* Call the :py:meth:`run <population.Population.run>` method on the ``Population`` object, giving it your fitness function and (optionally) the maximum number of generations you want NEAT to run.

.. note::
   **What's Happening: The Evolution Loop**
   
   Each generation, NEAT:
   
   1. **Evaluate**: Calls your fitness function for all genomes
   2. **Speciate**: Groups similar genomes into species (protects innovation)
   3. **Select**: Identifies the fittest genomes in each species
   4. **Reproduce**: Creates offspring through crossover and mutation
   5. **Mutate**: Randomly modifies offspring (weights, add/remove nodes/connections)
   6. **Repeat**: Until fitness threshold reached or max generations
   
   This implements the NEAT algorithm's core principle: start simple and complexify as needed.

After these three things are completed, NEAT will run until either you reach the specified number of generations, or
at least one genome achieves the :ref:`fitness_threshold <fitness-threshold-label>` value you specified in your config file.

Getting the results
-------------------

Once the call to the population object's ``run`` method has returned, you can query the ``statistics`` member of the
population (a :py:class:`neat.statistics.StatisticsReporter <statistics.StatisticsReporter>` object) to get the best genome(s) seen during the run.
In this example, we take the 'winner' genome to be that returned by ``pop.statistics.best_genome()``.

Other information available from the default statistics object includes per-generation mean fitness, per-generation standard deviation of fitness,
and the best N genomes (with or without duplicates).

Visualizations
--------------

Functions are available in the `visualize module
<https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/visualize.py>`_ to plot the best
and average fitness vs. generation, plot the change in species vs. generation, and to show the structure
of a network described by a genome.

Example Source
--------------

NOTE: This page shows the source and configuration file for the current version of neat-python available on
GitHub.  If you are using the version 0.92 installed from PyPI, make sure you get the script and config file from
the `archived source for that release <https://github.com/CodeReclaimers/neat-python/releases/tag/v0.92>`_.

Here's the entire example:

.. literalinclude:: ../examples/xor/evolve-feedforward.py

and here is the associated config file:

.. literalinclude:: ../examples/xor/config-feedforward