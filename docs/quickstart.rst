Quick Start
===========

Get NEAT-Python running in 5 minutes.

Installation
------------

Install via pip:

.. code-block:: bash

   pip install neat-python

That's it! NEAT-Python has no dependencies beyond the Python standard library.

.. note::
   NEAT-Python supports Python 3.8-3.14 and PyPy3.

Minimal Example
---------------

This example evolves a neural network to solve XOR (exclusive-or) in about 15 lines of code.

First, create a configuration file named ``config-xor``:

.. code-block:: ini

   [NEAT]
   fitness_criterion     = max
   fitness_threshold     = 3.9
   pop_size              = 150
   reset_on_extinction   = False
   no_fitness_termination = False

   [DefaultGenome]
   activation_default      = sigmoid
   activation_mutate_rate  = 0.0
   activation_options      = sigmoid
   
   aggregation_default     = sum
   aggregation_mutate_rate = 0.0
   aggregation_options     = sum
   
   bias_init_mean          = 0.0
   bias_init_stdev         = 1.0
   bias_init_type          = gaussian
   bias_max_value          = 30.0
   bias_min_value          = -30.0
   bias_mutate_power       = 0.5
   bias_mutate_rate        = 0.7
   bias_replace_rate       = 0.1
   
   compatibility_disjoint_coefficient = 1.0
   compatibility_weight_coefficient   = 0.5
   
   conn_add_prob           = 0.5
   conn_delete_prob        = 0.5
   
   enabled_default         = True
   enabled_mutate_rate     = 0.01
   enabled_rate_to_true_add  = 0.0
   enabled_rate_to_false_add = 0.0
   
   feed_forward            = True
   initial_connection      = full_direct
   
   node_add_prob           = 0.2
   node_delete_prob        = 0.2
   
   num_hidden              = 0
   num_inputs              = 2
   num_outputs             = 1
   
   response_init_mean      = 1.0
   response_init_stdev     = 0.0
   response_init_type      = gaussian
   response_max_value      = 30.0
   response_min_value      = -30.0
   response_mutate_power   = 0.0
   response_mutate_rate    = 0.0
   response_replace_rate   = 0.0
   
   weight_init_mean        = 0.0
   weight_init_stdev       = 1.0
   weight_init_type        = gaussian
   weight_max_value        = 30
   weight_min_value        = -30
   weight_mutate_power     = 0.5
   weight_mutate_rate      = 0.8
   weight_replace_rate     = 0.1
   
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

Now create your Python script:

.. code-block:: python

   import neat

   # XOR test cases: input -> expected output
   xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
   xor_outputs = [(0.0,),    (1.0,),    (1.0,),    (0.0,)]

   def eval_genomes(genomes, config):
       """Fitness function: evaluates how well each genome solves XOR."""
       for genome_id, genome in genomes:
           # Create a neural network from this genome
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           
           # Start with perfect fitness, subtract error
           genome.fitness = 4.0
           
           # Test on all 4 XOR cases
           for xi, xo in zip(xor_inputs, xor_outputs):
               output = net.activate(xi)
               genome.fitness -= (output[0] - xo[0]) ** 2

   # Load configuration
   config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-xor')

   # Create population
   p = neat.Population(config)
   p.add_reporter(neat.StdOutReporter(True))

   # Run evolution for up to 300 generations
   winner = p.run(eval_genomes, 300)

   # Test the winner
   print('\nBest genome:\n{!s}'.format(winner))

Run It!
-------

Save both files in the same directory and run:

.. code-block:: bash

   python your_script.py

You'll see generation-by-generation progress:

.. code-block:: text

   ****** Running generation 0 ******

   Population's average fitness: 2.33450 stdev: 0.52901
   Best fitness: 3.58932 - size: (2, 3) - species 1 - id 8
   Average adjusted fitness: 0.291
   Mean genetic distance 1.318, standard deviation 0.543
   Population of 150 members in 3 species:
      ID   age  size  fitness  adj fit  stag
     ====  ===  ====  =======  =======  ====
        1    0    86    3.589    0.306     0
        2    0    32    3.013    0.289     0
        3    0    32    2.945    0.279     0
   Total extinctions: 0
   Generation time: 0.024 sec

   ****** Running generation 50 ******
   
   Population's average fitness: 3.89234 stdev: 0.08234
   Best fitness: 3.98765 - size: (3, 5) - species 1 - id 1234

   [Evolution continues...]

   ****** Running generation 73 ******
   
   Best fitness: 3.96234 - size: (2, 4) - species 1 - id 2891

   Best individual in generation 73 meets fitness threshold - complexity: (2, 4)

   Best genome:
   Key: 2891
   Fitness: 3.96234
   Nodes:
       0 DefaultNodeGene(key=0, bias=-0.523, response=1.0, ...
   Connections:
       DefaultConnectionGene(key=(-1, 0), weight=4.832, enabled=True)
       DefaultConnectionGene(key=(-2, 0), weight=4.651, enabled=True)

The evolution typically converges in 50-150 generations and takes 1-3 seconds on modern hardware.

What Just Happened?
-------------------

Let's break down what NEAT did:

**1. Population Initialization**
   NEAT created 150 random neural networks (genomes), each starting simple with just inputs and outputs directly connected.

**2. Fitness Evaluation**
   Your ``eval_genomes`` function tested each network on all 4 XOR cases and assigned a fitness score. Better performance = higher fitness.

**3. Evolution Loop**
   Each generation, NEAT:
   
   - Groups similar genomes into species (protects innovation)
   - Selects the fittest individuals for reproduction
   - Creates the next generation through crossover and mutation
   - Sometimes adds nodes or connections (complexification)

**4. Termination**
   Evolution stops when a genome achieves fitness ≥ 3.9 (our threshold) or reaches 300 generations.

**5. Winner**
   The best genome is returned - a neural network that successfully computes XOR!

Key Concepts
~~~~~~~~~~~~

**Fitness Function**: You define how to measure success. NEAT handles everything else.

**Genome**: The genetic encoding of a neural network (nodes + connections + weights).

**Species**: Groups of similar genomes. This protects innovative structures from being eliminated too quickly.

**Complexification**: Networks start simple and add complexity only when beneficial.

Next Steps
----------

Now that you've seen NEAT in action, here's where to go next:

**Understand the Details**
   :doc:`xor_example` - Detailed walkthrough of the XOR example with visualizations
   
**Configure Your Evolution**
   :doc:`config_essentials` - Learn the key configuration parameters
   
   :doc:`config_file` - Complete configuration reference

**Understand NEAT**
   :doc:`neat_overview` - How the NEAT algorithm works
   
   :doc:`innovation_numbers` - Understanding v1.0.0's key feature

**Customize Behavior**
   :doc:`customization` - Advanced customization options
   
   :doc:`module_summaries` - API reference

**Examples**
   Check the ``examples/`` directory in the source repository for more:
   
   - Single pole balancing (control problem)
   - OpenAI Gym integration
   - Picture evolution
   - Memory tasks

Need Help?
----------

- **Documentation**: You're reading it! Use the navigation on the left.
- **GitHub Issues**: `Report bugs or ask questions <https://github.com/CodeReclaimers/neat-python/issues>`_
- **Source Code**: `Browse the repository <https://github.com/CodeReclaimers/neat-python>`_
