Understanding Innovation Numbers
================================

Innovation numbers are a key feature of the NEAT algorithm, and NEAT-Python v1.0.0 implements them exactly as described in the original paper. This page explains what they are, why they matter, and how v1.0.0's implementation improves correctness.

What Are Innovation Numbers?
-----------------------------

Innovation numbers are **unique historical markers** assigned to genes (specifically, connection genes). Each time a new connection appears through mutation, it receives a new innovation number that permanently identifies it.

Think of innovation numbers like serial numbers: once assigned, they never change and uniquely identify that particular genetic innovation across all generations.

A Simple Example
~~~~~~~~~~~~~~~~~

Imagine two genomes independently evolve a connection from neuron A to neuron B in the same generation:

.. code-block:: text

   Generation 5:
   
   Genome #42 mutates: adds connection A → B
   ├─ Innovation number 157 assigned
   
   Genome #87 mutates: adds same connection A → B
   └─ Gets same innovation number: 157 (tracked within generation)

   When #42 and #87 mate later:
   └─ Both have gene with innovation #157
   └─ NEAT knows these are the SAME gene
   └─ Crossover lines them up correctly ✓

Without innovation numbers:

.. code-block:: text

   Without tracking:
   
   When #42 and #87 mate:
   ├─ Both have connection A → B
   ├─ But are they the same gene or different?
   ├─ Crossover treats them as different genes ✗
   └─ Wrong offspring produced ✗

Why They Matter
---------------

Innovation numbers enable two critical features of NEAT:

1. **Correct Crossover**
   When two genomes mate, their genes must be properly aligned. Innovation numbers tell NEAT which genes in the two parents correspond to the same historical mutation, even if the genomes have very different structures.

2. **Accurate Speciation**
   NEAT groups similar genomes into species to protect innovation. It measures genetic distance by comparing which genes genomes share. Innovation numbers make this comparison accurate and efficient.

From the Original Paper
~~~~~~~~~~~~~~~~~~~~~~~~

Stanley & Miikkulainen (2002) explain:

   *"By keeping a list of the innovations that occurred in the current generation, it is possible to ensure that when the same structure arises more than once through independent mutations in the same generation, each identical mutation is assigned the same innovation number."*

This ensures that genomes that independently discover the same good mutation can properly exchange genetic material when they mate.

What Changed in v1.0.0
-----------------------

NEAT-Python v1.0.0 is a **major upgrade** that fully implements innovation number tracking per the original NEAT paper.

Before v1.0.0 (Incomplete Implementation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # v0.x: Connections identified only by endpoint nodes
   connection = DefaultConnectionGene(key=(-1, 0))  # Just (input, output)
   
   # No historical tracking
   # Same mutation in different genomes treated as different genes
   # Crossover based only on node pairs

**Limitations:**
- Crossover was less accurate
- Speciation calculations could be imprecise
- Not fully compliant with NEAT paper specification

v1.0.0 and Later (Full Implementation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # v1.0.0: Connections require innovation numbers
   connection = DefaultConnectionGene(
       key=(-1, 0),
       innovation=157  # Historical marker - MANDATORY
   )
   
   # Same mutation in same generation = same innovation number
   # Crossover matches genes by innovation number
   # Fully paper-compliant

**Benefits:**
- ✅ Correct crossover alignment
- ✅ Accurate speciation
- ✅ Full NEAT paper compliance
- ✅ Better evolution results

Breaking Change: Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Important**: Checkpoints saved with v0.x are **incompatible** with v1.0.0+.

The innovation tracking system requires additional data that wasn't stored in old checkpoints. If you need to continue old evolution runs, use v0.93 to load them and run to completion.

.. warning::
   Attempting to load a pre-v1.0 checkpoint will fail with an error. This is expected and by design.

Impact on Your Code
-------------------

Good News for Most Users
~~~~~~~~~~~~~~~~~~~~~~~~~

**If you're using NEAT-Python as a standard library**, innovation numbers are completely transparent. You don't need to change anything!

The innovation tracking system:
- ✅ Automatically initializes when you create a population
- ✅ Automatically assigns innovation numbers during mutation
- ✅ Automatically uses them during crossover
- ✅ Automatically saves/restores with checkpoints

Your existing code continues to work:

.. code-block:: python

   # This code works identically in v1.0.0
   config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-file')
   
   p = neat.Population(config)
   winner = p.run(eval_genomes, 300)
   
   # Innovation numbers handled automatically behind the scenes

Custom Genome Types
~~~~~~~~~~~~~~~~~~~

**If you've created custom genome classes** that inherit from ``DefaultGenome`` or implement the genome interface, you'll need to update them to support innovation numbers.

Key changes:
1. Connection genes must have an ``innovation`` parameter
2. Mutation methods must use the innovation tracker
3. Crossover must match genes by innovation number

See the :doc:`migration` guide for detailed instructions on updating custom genomes.

How It Works (Technical Details)
---------------------------------

.. note::
   This section is for advanced users who want to understand the implementation. It's not required knowledge for using NEAT-Python.

The Innovation Tracker
~~~~~~~~~~~~~~~~~~~~~~~

The ``InnovationTracker`` class (in ``neat/innovation.py``) manages innovation number assignment:

.. code-block:: python

   class InnovationTracker:
       def __init__(self):
           self.global_counter = 0  # Never resets
           self.generation_innovations = {}  # Resets each generation
       
       def get_innovation_number(self, input_node, output_node, mutation_type):
           """Get or assign innovation number for a connection."""
           key = (input_node, output_node, mutation_type)
           
           # Same mutation this generation? Return existing number
           if key in self.generation_innovations:
               return self.generation_innovations[key]
           
           # New mutation: assign next number
           self.global_counter += 1
           self.generation_innovations[key] = self.global_counter
           return self.global_counter

Mutation Types
~~~~~~~~~~~~~~

The tracker distinguishes between different types of structural mutations:

- ``'add_connection'``: New connection between existing nodes
- ``'add_node_in'``: Connection TO new node (when splitting existing connection)
- ``'add_node_out'``: Connection FROM new node (when splitting existing connection)
- ``'initial_connection'``: Connections in the initial population

This ensures that adding connection A→B gets a different innovation number than splitting a connection by adding a node.

Crossover by Innovation Number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When two genomes mate, genes are matched by innovation number:

.. code-block:: python

   def configure_crossover(parent1, parent2):
       # Build mappings: innovation_number → gene
       parent1_genes = {gene.innovation: gene for gene in parent1.connections.values()}
       parent2_genes = {gene.innovation: gene for gene in parent2.connections.values()}
       
       child_genes = {}
       
       # For each innovation number
       all_innovations = set(parent1_genes.keys()) | set(parent2_genes.keys())
       for innovation in all_innovations:
           if innovation in parent1_genes and innovation in parent2_genes:
               # Matching gene - randomly choose parent
               child_genes[innovation] = random.choice([
                   parent1_genes[innovation],
                   parent2_genes[innovation]
               ]).copy()
           elif fittest parent has it:
               # Disjoint/excess - inherit from fittest
               child_genes[innovation] = fittest_gene.copy()

Generation Tracking
~~~~~~~~~~~~~~~~~~~

At the start of each generation, the tracker resets its generation-specific record:

.. code-block:: python

   # In reproduction.py, at start of each generation:
   self.innovation_tracker.reset_generation()
   
   # This clears generation_innovations dict
   # But global_counter continues incrementing forever

This ensures:
- Same mutation in generation N gets same innovation number
- Same mutation in generation N+1 gets a DIFFERENT innovation number (as it should)
- Innovation numbers never reset or get reused

Checkpoint Serialization
~~~~~~~~~~~~~~~~~~~~~~~~~

The innovation tracker is automatically saved with checkpoints:

.. code-block:: python

   # Saving
   checkpoint_data = (generation, config, population, species_set, rndstate)
   # innovation_tracker saved as part of population.reproduction
   
   # Loading
   population = neat.Checkpointer.restore_checkpoint('checkpoint-file')
   # innovation_tracker automatically reconnected to genome_config

The global counter state is preserved, so innovation numbers continue from where they left off.

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

For complete implementation details, see ``INNOVATION_TRACKING_IMPLEMENTATION.md`` in the `source repository <https://github.com/CodeReclaimers/neat-python/blob/master/INNOVATION_TRACKING_IMPLEMENTATION.md>`_.

References
----------

Academic Paper
~~~~~~~~~~~~~~

**Stanley, K. O., & Miikkulainen, R. (2002).** *Evolving neural networks through augmenting topologies.* Evolutionary computation, 10(2), 99-127.

Section 3.2 (p. 108) describes innovation numbers:
   *"Whenever a new gene appears (through structural mutation), a global innovation number is incremented and assigned to that gene... By keeping a list of the innovations that occurred in the current generation, it is possible to ensure that when the same structure arises more than once through independent mutations in the same generation, each identical mutation is assigned the same innovation number."*

Paper available at: http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf

Related Documentation
~~~~~~~~~~~~~~~~~~~~~

- :doc:`migration` - Upgrading from v0.x to v1.0.0
- :doc:`neat_overview` - How NEAT works
- :doc:`customization` - Implementing custom genome types
- :doc:`glossary` - NEAT terminology

For questions or issues, please visit the `GitHub repository <https://github.com/CodeReclaimers/neat-python>`_.
