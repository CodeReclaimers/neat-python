
Module summaries
==================

TODO: Finish putting in modules; add links; fix problem with width of displayed linked highlighted source code.

.. py:module:: activations
   :synopsis: Has the built-in activation functions and code for using them and adding new user-defined ones.

activations
---------------

  .. py:exception:: InvalidActivationFunction(Exception)

                          Exception called if an activation function being added is invalid according to its `validate_activation` method.

  .. py:class:: ActivationFunctionSet

               Contains the list of current valid activation functions, including methods for adding and getting them.

.. py:module:: attributes
   :synopsis: Deals with attributes used by genes.

attributes
-------------

  .. inheritance-diagram:: attributes

  .. py:class:: BaseAttribute(self, name)

               Superclass for the type-specialized attribute subclasses.

  .. py:class:: FloatAttribute(BaseAttribute)

               Class for numeric attributes; includes code for configuration, creation, and mutation.

  .. py:class:: BoolAttribute(BaseAttribute)

               Class for boolean attributes; includes code for configuration, creation, and mutation.

  .. py:class:: StringAttribute(BaseAttribute)

               Class for string attributes, which are selected from a list of options; includes code for configuration, creation, and mutation.

.. py:module:: checkpoint
   :synopsis: Uses `pickle` to save and restore populations (and other aspects of the simulation state).

checkpoint
---------------

  .. py:class:: Checkpointer(self, generation_interval=100, time_interval_seconds=300)

               A reporter class that performs checkpointing using `pickle` to save and restore populations (and other aspects of the simulation state).

.. py:module:: config
   :synopsis: Does general configuration parsing; used by other classes for their configuration.

config
--------

  .. py:class:: ConfigParameter(self, name, value_type)

               Does initial handling of a particular configuration parameter.

               :param str name: The name of the configuration parameter.
               :param str value_type: The type that the configuration parameter should be; must be one of `str`, `int`, `bool`, `float`, or `list`.

  .. py:class:: Config(self, genome_type, reproduction_type, species_set_type, stagnation_type, filename)

               A simple container for user-configurable parameters of NEAT. The four parameters ending in ``_type`` may be the defaults or user-provided objects, which must make available the methods `parse_config` and `write_config`, plus others depending on which object it is.

.. py:module:: ctrnn
   :synopsis: Handles the continuous-time recurrent neural network implementation.

ctrnn
-------

  .. py:class:: CTRNNNodeEval(self, time_constant, activation, aggregation, bias, response, links)

		Sets up the basic ctrnn nodes.

  .. py:class:: CTRNN(self, inputs, outputs, node_evals)

               Sets up the ctrnn network itself.

.. py:module:: genes
   :synopsis: Handles node and connection genes.

genes
--------

  .. inheritance-diagram:: neat.genes iznn.IZNodeGene

  .. py:class:: BaseGene(self, key)

               Handles functions shared by multiple types of genes (both node and connection), including crossover and calling mutation methods.

               :param int key: The gene identifier. **For connection genes, genetic distances use the identifiers of the connected nodes, not the connection gene's identifier.**

  .. py:class:: DefaultNodeGene(BaseGene)

               Groups attributes specific to node genes (of the most frequently-used type) and calculates genetic distances between two homologous (not disjoint or excess) node genes.

  .. py:class:: DefaultConnectionGene(BaseGene)

               Groups attributes specific to connection genes and calculates genetic distances between two homologous (not disjoint or excess) connection genes.

.. py:module:: genome
   :synopsis: Handles genomes (individuals in the population).

genome
-----------

  .. inheritance-diagram:: neat.genome iznn.IZGenome

  .. py:class:: DefaultGenomeConfig(self, params)

               Does the configuration for the DefaultGenome class.

  .. py:class:: DefaultGenome(self, key)

               The provided genome class.

.. py:module:: graphs
   :synopsis: Directed graph algorithm implementations.

graphs
---------

  .. py:function:: creates_cycle(connections, test)

                   Returns true if the addition of the "test" connection would create a cycle, assuming that no cycle already exists in the graph represented by "connections". Used to avoid recurrent networks when a pure feed-forward network is desired.

  .. py:function:: required_for_output(inputs, outputs, connections)

		    Collect the nodes whose state is required to compute the final network output(s).

                    :param inputs: the input identifiers; **it is assumed that the input identifier set and the node identifier set are disjoint.**
                    :type inputs: list(int)
                    :param outputs: the output node identifiers; by convention, the output node ids are always the same as the output index.
                    :type outputs: list(int)
                    :param connections: list of (input, output) connections in the network; should only include enabled ones.
                    :type connections: list(list(int, int))
                    :returns: A list of layers, with each layer consisting of a set of identifiers.
                    :rtype: list(set(int))

  .. py:function:: feed_forward_layers(inputs, outputs, connections)

                   Collect the layers whose members can be evaluated in parallel in a feed-forward network.

                   :param inputs: the network input nodes.
                   :type inputs: list(int)
                   :param outputs: the output node identifiers.
                   :type outputs: list(int)
                   :param connections: list of (input, output) connections in the network; should only include enabled ones.
                   :type connections: list(list(int, int))
                   :returns: A list of layers, with each layer consisting of a set of node identifiers; only includes nodes returned by required_for_output.
                   :rtype: list(set(int))

.. py:module:: indexer
   :synopsis: Contains the Indexer class, to help with creating new identifiers/keys.

indexer
----------

  .. py:class:: Indexer(self, first)

                    Initializes an Indexer instance with the internal ID counter set to `first`.

                    :param int first: The initial identifier (key) to be used.

          .. py:method:: get_next(self, result=None)

                                If `result` is not None, then we return it unmodified.  Otherwise, we return the next ID and increment our internal counter.

                                :param result: Returned unmodified unless `None`.
                                :type result: int or None
                                :returns int: Identifier/key to use.

.. py:module:: iznn
   :synopsis: Implements a spiking neural network (closer to in vivo neural networks) based on Izhikevich's 2003 model.

iznn
------

  .. inheritance-diagram:: iznn

  .. py:class:: IZNodeGene(BaseGene)

                Contains attributes for the iznn node genes and determines genomic distances.

  .. py:class:: IZGenome(DefaultGenome)

               Contains the parse_config class method for iznn genome configuration.

  .. py:class:: IZNeuron(self, bias, a, b, c, d, inputs)

               Sets up and simulates the iznn nodes (neurons). TODO: Currently has some numerical stability problems; the time-step should be adjustable.

               :param float bias: The bias of the neuron.
               :param float a: The time scale of the recovery variable.
               :param float b: The sensitivity of the recovery variable.
               :param float c: The after-spike reset value of the membrane potential.
               :param float d: The after-spike reset of the recovery variable.
               :param inputs: A list of (input key, weight) pairs for incoming connections.
               :type inputs: list(list(int, float))

  .. py:class:: IZNN(self, neurons, inputs, outputs)

               Sets up the network itself and simulates it using the connections and neurons.

               :param list neurons: The IZNeuron instances needed.
               :param inputs: The input keys.
               :type inputs: list(int)
               :param outputs: The output keys.
               :type outputs: list(int)

               .. py:staticmethod:: create(genome, config)

                  Receives a genome and returns its phenotype (a neural network).

                  :returns object: An IZNN instance.

.. py:module:: math_util
   :synopsis: Contains some mathematical functions not found in the Python2 standard library, plus a mechanism for looking up some commonly used functions by name.

math_util
-------------

  .. py:data:: stat_functions

                  Lookup table for commonly used ``{value} -> value`` functions; includes `max`, `min`, `mean`, and `median`.

  .. py:function:: mean(values)

                       Returns the arithmetic mean.

  .. py:function:: median(values)

                       Returns the median. (Note: Does not average between middle values.)

  .. py:function:: variance(values)

                       Returns the variance.

  .. py:function:: stdev(values)

                       Returns the standard deviation. *Note spelling.*

  .. py:function:: softmax(values)

                        Compute the softmax (a differentiable/smooth approximization of the maximum function) of the given value set.
                        The softmax is defined as follows: :math:`\begin{equation}v_i = \exp(v_i) / s \text{, where } s = \sum(\exp(v_0), \exp(v_1), \dotsc)\end{equation}`.

.. py:module:: nn.feed_forward
   :synopsis: A straightforward feed-forward neural network NEAT implementation.

nn.feed_forward
----------------------

  .. py:class:: FeedForwardNetwork(self, inputs, outputs, node_evals)

      A straightforward feed-forward neural network NEAT implementation.

      :param inputs: The input keys (IDs).
      :type inputs: list(int)
      :param outputs: The output keys.
      :type outputs: list(int)
      :param node_evals: A list of node descriptions, with each node represented by a list.
      :type node_evals: list(list(object))

      .. py:staticmethod:: create(genome, config)

                                  Receives a genome and returns its phenotype (a `FeedForwardNetwork`).

.. py:module:: nn.recurrent
   :synopsis: A recurrent (but otherwise straightforward) neural network NEAT implementation.

nn.recurrent
----------------------

  .. py:class:: RecurrentNetwork(self, inputs, outputs, node_evals)

      A recurrent (but otherwise straightforward) neural network NEAT implementation.

      :param inputs: The input keys (IDs).
      :type inputs: list(int)
      :param outputs: The output keys.
      :type outputs: list(int)
      :param node_evals: A list of node descriptions, with each node represented by a list.
      :type node_evals: list(list(object))

      .. py:staticmethod:: create(genome, config)

                                  Receives a genome and returns its phenotype (a `RecurrentNetwork`).


.. Wanting to put in reporting.py next due to checkpoint.

.. py:module:: reporting
   :synopsis: Makes possible reporter classes, which are triggered on particular events and may provide information to the user, may do something else such as checkpointing, or may do both.

reporting
-----------

  .. inheritance-diagram:: neat.reporting checkpoint.Checkpointer

  .. py:class:: ReporterSet

                    Keeps track of the set of reporters and gives functions to dispatch them at appropriate points.

  .. py:class:: BaseReporter

                    Definition of the reporter interface expected by ReporterSet. Inheriting from it will provide a set of `dummy` methods to be overridden as desired.

  .. py:class:: StdOutReporter(self, show_species_detail)

                    Uses print to output information about the run; an example reporter class.

                    :param bool show_species_detail: Whether or not to show additional details about each species in the population.
