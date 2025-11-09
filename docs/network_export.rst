Network Export
==============

.. versionadded:: 1.0

NEAT-Python provides the ability to export trained neural networks to a framework-agnostic JSON format. This allows you to:

* Convert networks to other formats (ONNX, TensorFlow, PyTorch, etc.) using third-party tools
* Inspect and debug network structure in a human-readable format
* Share networks across platforms and programming languages
* Archive trained networks independently of NEAT-Python

The JSON format is designed to be self-contained and well-documented, making it easy for third parties to create converters to other frameworks without requiring changes to NEAT-Python itself.

.. contents::
   :local:
   :depth: 2

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

After training a network, export it using the :py:func:`export_network_json` function:

.. code-block:: python

   import neat
   from neat.export import export_network_json

   # After training...
   config = neat.Config(...)
   winner = population.run(eval_genomes, 300)
   
   # Create network from winner genome
   net = neat.nn.FeedForwardNetwork.create(winner, config)
   
   # Export to JSON file
   export_network_json(
       net,
       filepath='my_network.json',
       metadata={
           'fitness': winner.fitness,
           'generation': population.generation,
           'genome_id': winner.key
       }
   )

Export to String
~~~~~~~~~~~~~~~~

You can also export to a JSON string without writing to a file:

.. code-block:: python

   json_str = export_network_json(net, metadata={'fitness': winner.fitness})
   print(json_str)

Supported Network Types
-----------------------

All four NEAT-Python network types are supported:

* **FeedForwardNetwork** - Feed-forward networks with no cycles
* **RecurrentNetwork** - Networks that allow recurrent connections
* **CTRNN** - Continuous-Time Recurrent Neural Networks with time constants
* **IZNN** - Izhikevich Spiking Neural Networks

Each network type exports with its specific parameters preserved in the JSON format.

JSON Format Overview
--------------------

The exported JSON contains:

* **format_version** - Format version for compatibility tracking (currently "1.0")
* **network_type** - Type of network (feedforward, recurrent, ctrnn, or iznn)
* **metadata** - Optional information like fitness, generation, custom fields
* **topology** - Input/output structure (number and IDs)
* **nodes** - List of all nodes with activation functions, biases, etc.
* **connections** - List of weighted connections between nodes

Example Export
~~~~~~~~~~~~~~

Here's an example of exported JSON for a simple XOR network:

.. code-block:: json

   {
     "format_version": "1.0",
     "network_type": "feedforward",
     "metadata": {
       "created_timestamp": "2025-11-09T15:30:00Z",
       "neat_python_version": "1.0.0",
       "fitness": 3.95,
       "generation": 150
     },
     "topology": {
       "num_inputs": 2,
       "num_outputs": 1,
       "input_keys": [-1, -2],
       "output_keys": [0]
     },
     "nodes": [
       {
         "id": 0,
         "type": "output",
         "activation": {"name": "sigmoid", "custom": false},
         "aggregation": {"name": "sum", "custom": false},
         "bias": -0.123,
         "response": 1.0
       }
     ],
     "connections": [
       {"from": -1, "to": 0, "weight": 0.75, "enabled": true}
     ]
   }

Custom Functions
~~~~~~~~~~~~~~~~

If your network uses custom activation or aggregation functions, they will be flagged in the export:

.. code-block:: json

   {
     "id": 0,
     "activation": {"name": "my_custom_activation", "custom": true}
   }

Third-party converters should handle custom functions appropriately (error, warn, or approximate).

Complete Format Specification
------------------------------

For the complete JSON format specification, including:

* Detailed schema for all fields
* Examples for each network type (FeedForward, Recurrent, CTRNN, IZNN)
* Built-in activation and aggregation function reference
* Guidance for creating converters to other formats

See the `network-json-format.md <https://github.com/CodeReclaimers/neat-python/blob/master/docs/network-json-format.md>`_ document in the repository.

API Reference
-------------

.. py:module:: neat.export
   :synopsis: Network export functionality

.. py:function:: export_network_json(network, filepath=None, metadata=None)

   Export a NEAT network to JSON format.
   
   This function supports all NEAT network types (FeedForwardNetwork, RecurrentNetwork, CTRNN, IZNN).
   The exported JSON format is framework-agnostic and designed for conversion to other formats
   by third-party tools.
   
   :param network: A NEAT network instance to export
   :type network: FeedForwardNetwork or RecurrentNetwork or CTRNN or IZNN
   :param filepath: Optional path to write JSON file. If None, returns JSON string only.
   :type filepath: str or None
   :param metadata: Optional dict with additional information to include in export.
                    Common fields: 'fitness', 'generation', 'genome_id'. Custom fields are supported.
   :type metadata: dict or None
   :return: JSON string representation of the network (always returned, even when filepath is provided)
   :rtype: str
   :raises TypeError: If network is None or not a valid network instance
   :raises ValueError: If network type is not supported
   
   **Example:**
   
   .. code-block:: python
   
      from neat.export import export_network_json
      
      # Export to file with metadata
      json_str = export_network_json(
          network,
          filepath='network.json',
          metadata={
              'fitness': 98.5,
              'generation': 42,
              'genome_id': 123,
              'description': 'XOR solver'
          }
      )
      
      # Export to string only
      json_str = export_network_json(network)
      
      # Parse and use the JSON
      import json
      data = json.loads(json_str)
      print(f"Network type: {data['network_type']}")
      print(f"Number of nodes: {len(data['nodes'])}")

.. py:module:: neat.export.json_format
   :synopsis: JSON format validation and utilities

.. py:function:: validate_json(data)

   Validate the structure of exported JSON data.
   
   Checks that all required fields are present and properly structured according to
   the JSON schema specification.
   
   :param dict data: Dictionary containing parsed JSON data
   :return: True if valid
   :rtype: bool
   :raises ValueError: If data doesn't match expected schema

.. py:function:: is_builtin_activation(func)

   Check if an activation function is built-in to NEAT-Python.
   
   :param func: Activation function to check
   :return: True if function is from neat.activations module
   :rtype: bool

.. py:function:: is_builtin_aggregation(func)

   Check if an aggregation function is built-in to NEAT-Python.
   
   :param func: Aggregation function to check
   :return: True if function is from neat.aggregations module
   :rtype: bool

Converting to Other Formats
----------------------------

The JSON export is designed as an intermediate format. While NEAT-Python does not include converters
to specific frameworks (to avoid dependency bloat), the JSON format is well-documented to enable
third-party tools.

Suggested Workflow
~~~~~~~~~~~~~~~~~~

1. **Export from NEAT-Python** to JSON using :py:func:`export_network_json`
2. **Use a converter tool** (community-maintained or custom) to transform JSON to your target format
3. **Deploy** the network in your production framework

Creating a Converter
~~~~~~~~~~~~~~~~~~~~

When creating a converter to another format (ONNX, TensorFlow, PyTorch, etc.), key considerations include:

**Activation Function Mapping**
   Most common activations (sigmoid, tanh, relu) have direct equivalents in popular frameworks.
   Custom activations may need to be composed from basic operations or approximated.

**Aggregation Functions**
   Standard aggregations (sum, product, max, min) map straightforwardly.
   The ``response`` parameter scales aggregated inputs before activation.

**Network Topology**
   Parse the ``nodes`` and ``connections`` arrays to reconstruct the graph structure.
   For recurrent networks, handle cycles appropriately for your target framework.

**CTRNN/IZNN Support**
   Continuous-time networks may require ODE solvers.
   Spiking networks may require specialized frameworks.

See the complete format documentation for detailed converter guidance.

Examples
--------

Export After Training
~~~~~~~~~~~~~~~~~~~~~

A complete example showing training and export:

.. code-block:: python

   import neat
   from neat.export import export_network_json
   
   # XOR problem setup
   xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
   xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]
   
   def eval_genomes(genomes, config):
       for genome_id, genome in genomes:
           genome.fitness = 4.0
           net = neat.nn.FeedForwardNetwork.create(genome, config)
           for xi, xo in zip(xor_inputs, xor_outputs):
               output = net.activate(xi)
               genome.fitness -= (output[0] - xo[0]) ** 2
   
   # Load config and run
   config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-feedforward')
   
   p = neat.Population(config)
   p.add_reporter(neat.StdOutReporter(True))
   winner = p.run(eval_genomes, 300)
   
   # Create and export winner network
   winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
   export_network_json(
       winner_net,
       filepath='xor_winner.json',
       metadata={
           'fitness': winner.fitness,
           'generation': p.generation,
           'genome_id': winner.key,
           'problem': 'XOR'
       }
   )
   
   print(f"Network exported! Fitness: {winner.fitness:.4f}")

The complete example is available in ``examples/export/export_example.py``.

Export Different Network Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from neat.export import export_network_json
   import neat
   
   # FeedForward
   ff_net = neat.nn.FeedForwardNetwork.create(genome, config)
   export_network_json(ff_net, 'feedforward.json')
   
   # Recurrent
   rnn = neat.nn.RecurrentNetwork.create(genome, config)
   export_network_json(rnn, 'recurrent.json')
   
   # CTRNN
   ctrnn = neat.ctrnn.CTRNN.create(genome, config, time_constant=1.0)
   export_network_json(ctrnn, 'ctrnn.json')
   
   # IZNN (requires IZGenome)
   iznn = neat.iznn.IZNN.create(iz_genome, iz_config)
   export_network_json(iznn, 'iznn.json')

Batch Export
~~~~~~~~~~~~

Export multiple networks from a population:

.. code-block:: python

   from neat.export import export_network_json
   import os
   
   # Export top 10 genomes
   sorted_genomes = sorted(population.items(), 
                          key=lambda x: x[1].fitness, 
                          reverse=True)[:10]
   
   os.makedirs('exported_networks', exist_ok=True)
   
   for genome_id, genome in sorted_genomes:
       net = neat.nn.FeedForwardNetwork.create(genome, config)
       export_network_json(
           net,
           filepath=f'exported_networks/network_{genome_id}.json',
           metadata={
               'genome_id': genome_id,
               'fitness': genome.fitness,
               'rank': sorted_genomes.index((genome_id, genome)) + 1
           }
       )

Design Philosophy
-----------------

The JSON export feature follows these design principles:

**No Dependencies**
   Uses only Python standard library (json, datetime, inspect).
   No ML framework dependencies to keep NEAT-Python lean.

**Framework-Agnostic**
   Not tied to any specific ML framework.
   Enables conversion to ONNX, TensorFlow, PyTorch, CoreML, etc.

**Community-Extensible**
   Well-documented format allows anyone to create converters.
   Prevents "flavor of the year" feature requests from bloating the library.

**Version-Controlled**
   Format includes version field for future evolution.
   Breaking changes will increment version with migration guidance.

**Human-Readable**
   JSON is easy to inspect, debug, and understand.
   Formatted with indentation for readability.

See Also
--------

* :doc:`xor_example` - Complete training example
* :doc:`module_summaries` - Other NEAT-Python modules
* :doc:`customization` - Custom activation/aggregation functions
* `network-json-format.md <https://github.com/CodeReclaimers/neat-python/blob/master/docs/network-json-format.md>`_ - Complete format specification
