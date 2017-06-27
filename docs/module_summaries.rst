
Module summaries
==================

.. default-role:: any

.. todo::

  Finish putting in all needed material from modules; add links; go over parameters as used in code to make sure are described correctly.

.. py:module:: activations
   :synopsis: Has the built-in activation functions (see :ref:`activation-functions-label`) and code for using them and adding new user-defined ones.

activations
---------------

  .. py:exception:: InvalidActivationFunction(Exception)

    Exception called if an activation function being added is invalid according to the `validate_activation` function.

  .. py:function:: validate_activation(function)

    Checks to make sure its parameter is a function that takes a single argument.

    :param object function: Object to be checked.
    :raises InvalidActivationFunction: If the object does not pass the tests.

  .. py:class:: ActivationFunctionSet

    Contains the list of current valid activation functions, including methods for adding and getting them.

.. py:module:: attributes
   :synopsis: Deals with :term:`attributes` used by genes.

attributes
-------------

  .. inheritance-diagram:: attributes

  .. py:class:: BaseAttribute(name)

    Superclass for the type-specialized attribute subclasses, used by genes (such as via the :py:class:`DefaultGene` implementation).

  .. py:class:: FloatAttribute(BaseAttribute)

    Class for numeric :term:`attributes` such as the :term:`response` of a :term:`node`; includes code for configuration, creation, and mutation.

  .. py:class:: BoolAttribute(BaseAttribute)

    Class for boolean :term:`attributes` such as whether a :term:`connection` is :term:`enabled` or not; includes code for configuration, creation, and mutation.

  .. py:class:: StringAttribute(BaseAttribute)

    Class for string attributes such as the :term:`aggregation function` of a :term:`node`, which are selected from a list of options;
    includes code for configuration, creation, and mutation.

.. py:module:: checkpoint
   :synopsis: Uses :py:mod:`pickle` to save and restore populations (and other aspects of the simulation state).

checkpoint
---------------

  .. py:class:: Checkpointer(generation_interval=100, time_interval_seconds=300)

    A reporter class that performs checkpointing using :py:mod:`pickle` to save and restore populations (and other aspects of the simulation state). It saves the
    current state every ``generation_interval`` generations or ``time_interval_seconds`` seconds, whichever happens first. Subclasses :py:class:`BaseReporter`.
    (The potential save point is at the end of a generation.)

    :param generation_interval: If not None, maximum number of generations between checkpoints.
    :type generation_interval: int or None
    :param time_interval_seconds: If not None, maximum number of seconds between checkpoints.
    :type time_interval_seconds: float or None

    .. py:staticmethod:: save_checkpoint(config, population, species, generation)

      Saves the current simulation (including randomization) state to :file:`neat-checkpoint-{generation}`, with ``generation`` being the generation number.

    .. py:staticmethod:: restore_checkpoint(filename)

      Resumes the simulation from a previous saved point. Loads the specified file, sets the randomization state, and returns a :py:class:`Population` object
      set up with the rest of the previous state.

      :param str filename: The file to be restored from.
      :return: Object that can be used with :py:meth:`Population.run <population.Population.run>` to restart the simulation.
      :rtype: :py:class:`Population <population.Population>` object.

.. todo:: Put in links to the customization page.

.. py:module:: config
   :synopsis: Does general configuration parsing; used by other classes for their configuration.

config
--------

  .. py:class:: ConfigParameter(name, value_type)

    Does initial handling of a particular configuration parameter.

    :param str name: The name of the configuration parameter.
    :param str value_type: The type that the configuration parameter should be; must be one of `str`, `int`, `bool`, `float`, or `list`.

  .. py:function:: write_pretty_params(f, config, params)

    Prints configuration parameters to `file` f.

  .. py:class:: Config(genome_type, reproduction_type, species_set_type, stagnation_type, filename)

    A simple container for user-configurable parameters of NEAT. The four parameters ending in ``_type`` may be the built-in ones or user-provided objects, which
    must make available the methods ``parse_config`` and ``write_config``, plus others depending on which object it is.
    ``Config`` itself takes care of the ``NEAT`` parameters. For a description of the configuration file, see :ref:`configuration-file-description-label`.

    :param object genome_type: Specifies the genome class used, such as :py:class:`DefaultGenome` or :py:class:`iznn.IZGenome`. See :ref:`genome-interface-label` for the needed interface.
    :param object reproduction_type: Specifies the reproduction class used, such as :py:class:`DefaultReproduction`. See :ref:`reproduction-interface-label` for the needed interface.
    :param object species_set_type: Specifies the species set class used, such as :py:class:`DefaultSpeciesSet`.
    :param object stagnation_type: Specifies the stagnation class used, such as :py:class:`DefaultStagnation`.
    :param str filename: Configuration file to be opened, read, processed by a parser from the :py:mod:`configparser` module, the ``NEAT`` section handled by ``Config``, and then other sections passed to the ``parse_config`` methods of the appropriate classes.
    :raises AssertionError: If any of the objects lack a ``parse_config`` method.

    .. py:method:: save(filename)

      Opens the specified file for writing (not appending) and outputs a configuration file from the current configuration. Uses :py:func:`write_pretty_params` for
      the ``NEAT`` parameters and the appropriate class ``write_config`` methods for the other sections.

      :param str filename: The configuration file to be written.

.. py:module:: ctrnn
   :synopsis: Handles the continuous-time recurrent neural network implementation.

ctrnn
-------

  .. py:class:: CTRNNNodeEval(time_constant, activation, aggregation, bias, response, links)

    Sets up the basic ctrnn nodes.

  .. py:class:: CTRNN(inputs, outputs, node_evals)

    Sets up the ctrnn network itself.

.. py:module:: genes
   :synopsis: Handles node and connection genes.

genes
--------

  .. inheritance-diagram:: neat.genes iznn.IZNodeGene

  .. py:class:: BaseGene(key)

    Handles functions shared by multiple types of genes (both node and connection), including crossover and calling mutation methods.

    :param int key: The gene identifier. **For connection genes, genetic distances use the identifiers of the connected nodes, not the connection gene's identifier.**

    .. py:classmethod:: parse_config(config, param_dict)

      Placeholder; parameters are entirely in gene attributes.

    .. py:classmethod:: get_config_params()

      Fetches configuration parameters from gene attributes.

  .. py:class:: DefaultNodeGene(BaseGene)

    Groups :py:mod:`attributes` specific to :term:`node` genes (of the usually-used type) and calculates genetic distances between two
    :term:`homologous` (not disjoint or excess) node genes.

    .. py:method:: distance(other, config)

      Determines weight of differences between node genes using their 4 :term:`attributes`; the final result is multiplied by the configured ``compatibility_weight_coefficient``.

      :param object other: The other ``DefaultNodeGene``.
      :param object config: The genome configuration object.
      :return: The contribution of this pair to the :term:`genomic distance` between the source genomes.
      :rtype: float

  .. py:class:: DefaultConnectionGene(BaseGene)

    Groups :py:mod:`attributes` specific to :term:`connection` genes and calculates genetic distances between two
    :term:`homologous` (not disjoint or excess) connection genes.

    .. py:method:: distance(other, config)

      Determines weight of differences between connection genes using their 2 :term:`attributes`;
      the final result is multiplied by the configured ``compatibility_weight_coefficient``.

      :param object other: The other ``DefaultConnectionGene``.
      :param object config: The genome configuration object.
      :return: The contribution of this pair to the :term:`genomic distance` between the source genomes.
      :rtype: float

.. todo::

   Explain more regarding parameters, required functions of the below; put in links referencing genome-interface.

.. py:module:: genome
   :synopsis: Handles genomes (individuals in the population).

genome
-----------

  .. inheritance-diagram:: neat.genome iznn.IZGenome

  .. py:class:: DefaultGenomeConfig(params)

    Does the configuration for the DefaultGenome class.

    :param dict params: Parameters from configuration file and DefaultGenome initialization (by parse_config).

  .. py:class:: DefaultGenome(key)

    The provided genome class. For class requirements, see :ref:`genome-interface-label`.

    :param int key: Identifier for this individual/genome.

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Provides default node and connection gene specifications (from :py:mod:`genes`) and uses `DefaultGenomeConfig` to
      do the rest of the configuration.

      :param dict param_dict: Dictionary of parameters from configuration file.
      :return: Configuration object; considered opaque by rest of code, so type may vary by implementation (here, a `DefaultGenomeConfig` instance).
      :rtype: object

    .. py:classmethod:: write_config(f, config)

      Required interface method. Saves configuration using `DefaultGenomeConfig`.

    .. py:method:: configure_new(config)

      Required interface method. Configure a new genome based on the given configuration.

    .. py:method:: configure_crossover(genome1, genome2, config)

      Required interface method. Configure a new genome by crossover from two parent genomes.

    .. py:method:: mutate(config)

      Required interface method. Mutates this genome.

    .. py:method:: distance(other, config)

      Required interface method. Returns the :term:`genomic distance` between this genome and the other. This distance value is used to compute
      genome compatibility for speciation. Uses the :py:meth:`DefaultNodeGene.distance` and :py:meth:`DefaultConnectionGene.distance` methods for
      :term:`homologous` pairs, and the configured ``compatibility_disjoint_coefficient`` for disjoint/excess genes.

      :param object other: The other DefaultGenome instance (genome) to be compared to.
      :param object config: The genome configuration object.
      :return: The genomic distance.
      :rtype: float

    .. py:method:: size()

      Required interface method. Returns genome ``complexity``, taken to be (number of nodes, number of enabled connections); currently only used
      for reporters - they are given this information for the highest-fitness genome at the end of each generation.

.. py:module:: graphs
   :synopsis: Directed graph algorithm implementations.

graphs
---------

  .. py:function:: creates_cycle(connections, test)

    Returns true if the addition of the ``test`` :term:`connection` would create a cycle, assuming that no cycle already exists in the graph represented by ``connections``.
    Used to avoid :term:`recurrent` networks when a purely :term:`feed-forward` network is desired (e.g., as determined by the ``feed_forward`` setting in the
    :ref:`configuration file <feed-forward-config-label>`.

    :param connections: The current network, as a list of (input, output) connections.
    :type connections: list(tuple(int, int))
    :param test: Possible connection to be checked for causing a cycle.
    :type test: tuple(int, int)
    :return: True if a cycle would be created; false if not.
    :rtype: bool

  .. py:function:: required_for_output(inputs, outputs, connections)

    Collect the nodes whose state is required to compute the final network output(s).

    :param inputs: the input identifiers; **it is assumed that the input identifier set and the node identifier set are disjoint.**
    :type inputs: list(int)
    :param outputs: the output node identifiers; by convention, the output node ids are always the same as the output index.
    :type outputs: list(int)
    :param connections: list of (input, output) connections in the network; should only include enabled ones.
    :type connections: list(tuple(int, int))
    :return: A list of layers, with each layer consisting of a set of node identifiers.
    :rtype: list(set(int))

  .. py:function:: feed_forward_layers(inputs, outputs, connections)

    Collect the layers whose members can be evaluated in parallel in a :term:`feed-forward` network.

    :param inputs: the network input nodes.
    :type inputs: list(int)
    :param outputs: the output node identifiers.
    :type outputs: list(int)
    :param connections: list of (input, output) connections in the network; should only include enabled ones.
    :type connections: list(tuple(int, int))
    :return: A list of layers, with each layer consisting of a set of identifiers; only includes nodes returned by required_for_output.
    :rtype: list(set(int))

.. py:module:: indexer
   :synopsis: Contains the Indexer class, to help with creating new identifiers/keys.

indexer
----------

  .. py:class:: Indexer(first)

    Initializes an Indexer instance with the internal ID counter set to ``first``.

    :param int first: The initial identifier (key) to be used.

    .. py:method:: get_next(result=None)

      If ``result`` is not `None`, then we return it unmodified.  Otherwise, we return the next ID and increment our internal counter.

      :param result: Returned unmodified unless `None`.
      :type result: int or None
      :return int: Identifier/key to use.

.. py:module:: iznn
   :synopsis: Implements a spiking neural network (closer to in vivo neural networks) based on Izhikevich's 2003 model.

iznn
------

  .. inheritance-diagram:: iznn

  .. py:class:: IZNodeGene(BaseGene)

    Contains attributes for the iznn node genes and determines genomic distances.

  .. py:class:: IZGenome(DefaultGenome)

    Contains the parse_config class method for iznn genome configuration.

  .. py:class:: IZNeuron(bias, a, b, c, d, inputs)

    Sets up and simulates the iznn nodes (neurons). TODO: Currently has some numerical stability problems; the time-step should be adjustable.

    :param float bias: The bias of the neuron.
    :param float a: The time scale of the recovery variable.
    :param float b: The sensitivity of the recovery variable.
    :param float c: The after-spike reset value of the membrane potential.
    :param float d: The after-spike reset of the recovery variable.
    :param inputs: A list of (input key, weight) pairs for incoming connections.
    :type inputs: list(tuple(int, float))

  .. py:class:: IZNN(neurons, inputs, outputs)

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

    Returns the median. (Note: For even numbers of values, does not take the mean between the two middle values.)

  .. py:function:: variance(values)

    Returns the (population) variance.

  .. py:function:: stdev(values)

    Returns the (population) standard deviation. *Note spelling.*

  .. py:function:: softmax(values)

    Compute the softmax (a differentiable/smooth approximization of the maximum function) of the given value set.
    The softmax is defined as follows: :math:`\begin{equation}v_i = \exp(v_i) / s \text{, where } s = \sum(\exp(v_0), \exp(v_1), \dotsc)\end{equation}`.

.. py:module:: nn.feed_forward
   :synopsis: A straightforward feed-forward neural network NEAT implementation.

nn.feed_forward
----------------------

  .. py:class:: FeedForwardNetwork(inputs, outputs, node_evals)

    A straightforward (no pun intended) feed-forward neural network NEAT implementation.

    :param inputs: The input keys (IDs).
    :type inputs: list(int)
    :param outputs: The output keys.
    :type outputs: list(int)
    :param node_evals: A list of node descriptions, with each node represented by a list.
    :type node_evals: list(list(object))

    .. py:staticmethod:: create(genome, config)

      Receives a genome and returns its phenotype (a :py:class:`FeedForwardNetwork`).

.. py:module:: nn.recurrent
   :synopsis: A recurrent (but otherwise straightforward) neural network NEAT implementation.

nn.recurrent
----------------------

  .. py:class:: RecurrentNetwork(inputs, outputs, node_evals)

    A recurrent (but otherwise straightforward) neural network NEAT implementation.

    :param inputs: The input keys (IDs).
    :type inputs: list(int)
    :param outputs: The output keys.
    :type outputs: list(int)
    :param node_evals: A list of node descriptions, with each node represented by a list.
    :type node_evals: list(list(object))

    .. py:staticmethod:: create(genome, config)

      Receives a genome and returns its phenotype (a :py:class:`RecurrentNetwork`).

.. py:module:: parallel
   :synopsis: Runs evaluation functions in parallel subprocesses in order to evaluate multiple genomes at once.

parallel
----------

  .. py:class:: ParallelEvaluator(num_workers, eval_function, timeout=None)

    Runs evaluation functions in parallel subprocesses in order to evaluate multiple genomes at once.

    :param int num_workers: How many workers to have in the :py:class:`Pool`.
    :param function eval_function: eval_function should take one argument (a genome object) and return a single float (the genome's fitness)
    :param timeout: How long (in seconds) each subprocess will be given before an exception is raised (unlimited if `None`).
    :type timeout: int or None

.. todo::

  Put in more about calls to rest of program?

.. index:: fitness function

.. py:module:: population
   :synopsis: Implements the core evolution algorithm.

population
--------------

  .. py:exception:: CompleteExtinctionException

    Raised on complete extinction (all species removed due to stagnation) unless :ref:`reset_on_extinction <reset-on-extinction-label>` is set.

  .. py:class:: Population(config, initial_state=None)

    This class implements the core evolution algorithm:
    1. Evaluate fitness of all genomes.
    2. Check to see if the termination criterion is satisfied; exit if it is.
    3. Generate the next generation from the current population.
    4. Partition the new generation into species based on genetic similarity.
    5. Go to 1.

    :param object config: The :py:class:`Config` configuration object.
    :param initial_state: If supplied (such as by a method of the :py:class:`Checkpointer` class), a tuple of (``Population``, ``Species``, generation number)
    :type initial_state: None or tuple(object, object, int)

    .. py:method:: run(fitness_function, n=None)

      Runs NEAT's genetic algorithm for at most n generations.  If n
      is ``None``, run until solution is found or extinction occurs.

      The user-provided fitness_function must take only two arguments:
      1. The population as a list of (genome id, genome) tuples.
      2. The current configuration object.

      The return value of the fitness function is ignored, but it must assign
      a Python `float` to the ``fitness`` member of each genome.

      The fitness function is free to maintain external state, perform
      evaluations in :py:mod::`parallel`, etc.

      It is assumed that the fitness function does not modify the list of genomes,
      the genomes themselves (apart from updating the fitness member),
      or the configuration object.

      :param object fitness_function: The fitness function to use, with arguments specified above.
      :param n: The maximum number of generations to run (unlimited if ``None``).
      :type n: int or None
      :return: The best genome seen.
      :rtype: object

.. todo::

  Put in reporter interface, clarify when called (e.g., ``found_solution`` is not called on reaching the max generation, only on satisfaction of the fitness criterion).

.. py:module:: reporting
   :synopsis: Makes possible reporter classes, which are triggered on particular events and may provide information to the user, may do something else such as checkpointing, or may do both.

reporting
-----------

  .. inheritance-diagram:: neat.reporting checkpoint.Checkpointer statistics.StatisticsReporter

  .. py:class:: ReporterSet

    Keeps track of the set of reporters and gives functions to dispatch them at appropriate points.

  .. py:class:: BaseReporter

    Definition of the reporter interface expected by ReporterSet. Inheriting from it will provide a set of ``dummy`` methods to be overridden as desired.

  .. py:class:: StdOutReporter(show_species_detail)

    Uses print to output information about the run; an example reporter class.

    :param bool show_species_detail: Whether or not to show additional details about each species in the population.

.. py:module:: reproduction
   :synopsis: Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents.

reproduction
-----------------

  .. py:class:: DefaultReproduction(config, reporters, stagnation)

    Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents. Implements the default NEAT-python reproduction
    scheme: explicit fitness sharing with fixed-time species stagnation. For class requirements, see :ref:`reproduction-interface-label`.

    :param dict config: Configuration object, in this case a dictionary.
    :param object reporters: A :py:class:`ReporterSet` object.
    :param object stagnation: A :py:class:`DefaultStagnation` object - current code partially depends on internals of this class (a TODO is noted to correct this)

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Provides defaults for ``elitism``, ``survival_threshold``, and ``min_species_size`` parameters and updates them from the configuration file.

      :params dict param_dict: Dictionary of parameters from configuration file.
      :return: Configuration object; considered opaque by rest of code, so current type returned is not required for interface.
      :rtype: dict

    .. py:classmethod:: write_config(f, param_dict)

      Required interface method. Saves ``elitism`` and ``survival_threshold`` (but not ``min_species_size``) parameters to new config file.

      :params file f: File to write to.
      :params dict param_dict: Dictionary of current parameters in this implementation; more generally, reproduction config object.

    .. py:method:: create_new(genome_type, genome_config, num_genomes)

      Required interface method. Creates ``num_genomes`` new genomes of the given type using the given configuration. Also initializes ancestry information (empty tuple).

      :param class genome_type: Genome class (such as :py:class:`DefaultGenome` or :py:class:`IZGenome`) to create instances of.
      :param object genome_config: Opaque genome configuration object.
      :param int num_genomes: How many new genomes to create.
      :return: A dictionary (with the unique genome identifier as the key) of the genomes created.
      :rtype: dict(int, object)

    .. py:method:: reproduce(config, species, pop_size, generation)

      Creates the population to be used in the next generation from the given configuration instance, SpeciesSet instance, desired size of the population, and
      current generation number.  This method is called after all genomes have been evaluated and their ``fitness`` member assigned.  This method
      should use the stagnation instance given to the initializer to remove species deemed to have stagnated.

      :param object config: A :py:class:`Config` instance.
      :param object species: A :py:class:`SpeciesSet` instance. As well as depending on some of the :py:class:`DefaultStagnation` internals, this method also depends on some of those of the species object.
      :param int pop_size: Population desired.
      :param int generation: Generation count.
      :return: New population, as a dict of unique genome ID/key vs genome.
      :rtype: dict

.. py:module:: six_util
   :synopsis: Provides Python 2/3 portability with three dictionary iterators; copied from the `six` module.

six_util
----------

This Python 2/3 portability code was copied from the `six module <https://pythonhosted.org/six/>`_ to avoid adding it as a dependency.

  .. py:function:: iterkeys(d, **kw)

    This function returns an iterator over the keys of dict d.

    :param dict d: Dictionary to iterate over
    :param kw: The function of this parameter is unclear.

  .. py:function:: iteritems(d, **kw)

    This function returns an iterator over the (key, value) pairs of dict d.

    :param dict d: Dictionary to iterate over
    :param kw: The function of this parameter is unclear.

  .. py:function:: itervalues(d, **kw)

    This function returns an iterator over the values of dict d.

    :param dict d: Dictionary to iterate over
    :param kw: The function of this parameter is unclear.

.. Internally, the above are using ``**kw`` as a PARAMETER for keys/items/values/iterkeys/iteritems/itervalues. ??? Is this in case someone puts in
.. a set of key/value pairs instead of a dictionary? The `six` documentation just states that this parameter is "passed to the underlying method", which is not helpful.

.. todo::

   Add at least some methods to the below for DefaultSpeciesSet; try to figure out which ones are required interface methods.

.. py:module:: species
   :synopsis: Divides the population into genome-based species.

species
-----------

  .. py:class:: Species(key, generation)

    Represents a species and contains data about it such as members, fitness, and time stagnating.

    :param int key: Identifier
    :param int generation: Initial generation of appearance

  .. py:class:: GenomeDistanceCache(config)

    Caches genomic distance information to avoid repeated lookups. Called as a method with a pair of genomes to retrieve the distance.

  .. py:class:: DefaultSpeciesSet(config, reporters)

    Encapsulates the default speciation scheme by configuring it and performing the speciation function (placing genomes into species by genetic similarity).

.. todo::

   Add more methods to the below for DefaultStagnation; try to figure out which ones are required interface methods.

.. py:module:: stagnation
   :synopsis: Keeps track of whether species are making progress and removes ones that are not (for a configurable number of generations).

stagnation
--------------

  .. py:class:: DefaultStagnation(config, reporters)

    Keeps track of whether species are making progress and helps remove ones that, for a configurable number of generations, are not.

    :param object config: Configuration object; in this implementation, a `dict`, but should be treated as opaque outside this class.
    :param class reporters: A :py:class:`ReporterSet` with reporters that may need activating; not currently used.

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Provides defaults for ``species_fitness_func``, ``max_stagnation``, and ``species_elitism`` parameters and updates them from the
      configuration file.

      :params dict param_dict: Dictionary of parameters from configuration file.
      :return: Configuration object; considered opaque by rest of code, so current type returned is not required for interface.
      :rtype: dict

    .. py:classmethod:: write_config(f, param_dict)

      Required interface method. Saves parameters to new config file. **Has a default of 15 for species_elitism, but will be overridden by the default of 0 in parse_config.**

      :params file f: File to write to.
      :params dict param_dict: Dictionary of current parameters in this implementation; more generally, stagnation config object.

.. py:module:: statistics
   :synopsis: Gathers and provides (to callers and/or to a file) information on genome and species fitness, which are the most-fit genomes, and similar.

statistics
-------------

  .. py:class:: StatisticsReporter(BaseReporter)

    Gathers (via the reporting interface) and provides (to callers and/or to a file) information on genome and species fitness, which are the most-fit genomes, etc.
    Note: Keeps accumulating information in memory currently, which may be a problem in long runs.
