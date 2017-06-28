
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

    Checks to make sure its parameter is a function that takes a single argument. TODO: Currently raises a deprecation warning due to changes in `inspect`.

    :param object function: Object to be checked.
    :raises InvalidActivationFunction: If the object does not pass the tests.

  .. py:class:: ActivationFunctionSet

    Contains the list of current valid activation functions, including methods for adding and getting them.

.. Suggested simplification for the below: Make __config_items__ a list of lists/tuples, with the latter containing (name, value_type, default) - no default if the last is None.
.. This would also allow moving get_config_params into the BaseAttribute class, although config_item_names may require some modifications.

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

    Prints configuration parameters to `file` object f.

  .. py:class:: Config(genome_type, reproduction_type, species_set_type, stagnation_type, filename)

    A simple container for user-configurable parameters of NEAT. The four parameters ending in ``_type`` may be the built-in ones or user-provided objects, which
    must make available the methods ``parse_config`` and ``write_config``, plus others depending on which object it is.
    ``Config`` itself takes care of the ``NEAT`` parameters. For a description of the configuration file, see :ref:`configuration-file-description-label`.

    :param object genome_type: Specifies the genome class used, such as :py:class:`DefaultGenome` or :py:class:`iznn.IZGenome`. See :ref:`genome-interface-label` for the needed interface.
    :param object reproduction_type: Specifies the reproduction class used, such as :py:class:`DefaultReproduction`. See :ref:`reproduction-interface-label` for the needed interface.
    :param object species_set_type: Specifies the species set class used, such as :py:class:`DefaultSpeciesSet`.
    :param object stagnation_type: Specifies the stagnation class used, such as :py:class:`DefaultStagnation`.
    :param str filename: Pathname for configuration file to be opened, read, processed by a parser from the :py:mod:`configparser` module, the ``NEAT`` section handled by ``Config``, and then other sections passed to the ``parse_config`` methods of the appropriate classes.
    :raises AssertionError: If any of the objects lack a ``parse_config`` method.

    .. py:method:: save(filename)

      Opens the specified file for writing (not appending) and outputs a configuration file from the current configuration. Uses :py:func:`write_pretty_params` for
      the ``NEAT`` parameters and the appropriate class ``write_config`` methods for the other sections.

      :param str filename: The configuration file to be written.

.. todo::

  Give more information about parameters for ctrnn.

.. py:module:: ctrnn
   :synopsis: Handles the continuous-time recurrent neural network implementation.

ctrnn
-------

  .. py:class:: CTRNNNodeEval(time_constant, activation, aggregation, bias, response, links)

    Sets up the basic :doc:`ctrnn` nodes.

  .. py:class:: CTRNN(inputs, outputs, node_evals)

    Sets up the :doc:`ctrnn` network itself.

    .. py:method:: reset()

      Resets the time and all node activations to 0 (necessary due to otherwise retaining state via recurrent connections).

    .. py:method:: advance(inputs, advance_time, time_step=None)

      Advance the simulation by the given amount of time, assuming that inputs are
      constant at the given values during the simulated time.

      :param list inputs: The values for the :term:`input nodes <input node>`.
      :param float advance_time: How much time to advance the network before returning the resulting outputs.
      :param float time_step: How much time per step to advance the network; the default of ``None`` will currently result in an error, but it is planned to determine it automatically.
      :return: The values for the :term:`output nodes <output node>`.
      :rtype: list

    .. py:staticmethod:: create(genome, config, time_constant)

      Receives a genome and returns its phenotype (a :py:class:`CTRNN`). The ``time_constant`` is used for the :py:class:`CTRNNNodeEval` initializations.

.. index:: ! genomic distance
.. index:: ! gene

.. py:module:: genes
   :synopsis: Handles node and connection genes.

genes
--------

  .. inheritance-diagram:: neat.genes iznn.IZNodeGene

  .. py:class:: BaseGene(key)

    Handles functions shared by multiple types of genes (both :term:`node` and :term:`connection`), including crossover and calling mutation methods.

    :param int key: The gene identifier. **For connection genes, determining whether they are homologous (for genomic distance determination) uses the identifiers of the connected nodes, not the connection gene's identifier.**

    .. py:classmethod:: parse_config(config, param_dict)

      Placeholder; parameters are entirely in gene attributes.

    .. py:classmethod:: get_config_params()

      Fetches configuration parameters from gene attributes.

  .. py:class:: DefaultNodeGene(BaseGene)

    Groups :py:mod:`attributes` specific to :term:`node` genes (of the usually-used type) and calculates genetic distances between two
    :term:`homologous` (not disjoint or excess) node genes.

    .. py:method:: distance(other, config)

      Determines weight of differences between node genes using their 4 :term:`attributes`;
      the final result is multiplied by the configured :ref:`compatibility_weight_coefficient <compatibility-weight-coefficient-label>`.

      :param object other: The other ``DefaultNodeGene``.
      :param object config: The genome configuration object.
      :return: The contribution of this pair to the :term:`genomic distance` between the source genomes.
      :rtype: float

  .. py:class:: DefaultConnectionGene(BaseGene)

    Groups :py:mod:`attributes` specific to :term:`connection` genes and calculates genetic distances between two
    :term:`homologous` (not disjoint or excess) connection genes.

    .. py:method:: distance(other, config)

      Determines weight of differences between connection genes using their 2 :term:`attributes`;
      the final result is multiplied by the configured :ref:`compatibility_weight_coefficient <compatibility-weight-coefficient-label>`.

      :param object other: The other ``DefaultConnectionGene``.
      :param object config: The genome configuration object.
      :return: The contribution of this pair to the :term:`genomic distance` between the source genomes.
      :rtype: float

.. todo::

   Explain more regarding parameters, required functions of the below.

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

      :param file f: File object to write to.
      :param object config: Configuration object (here, a `DefaultGenomeConfig` instance).

    .. py:method:: configure_new(config)

      Required interface method. Configures a new genome (itself) based on the given configuration object.

    .. py:method:: configure_crossover(genome1, genome2, config)

      Required interface method. Configures a new genome (itself) by crossover from two parent genomes.

    .. py:method:: mutate(config)

      Required interface method. Mutates this genome.

    .. py:method:: distance(other, config)

      Required interface method. Returns the :term:`genomic distance` between this genome and the other. This :index:`distance <single: genomic distance>`
      value is used to compute genome compatibility for :py:mod:`speciation <species>`. Uses the
      :py:meth:`DefaultNodeGene.distance` and :py:meth:`DefaultConnectionGene.distance` methods for
      :term:`homologous` pairs, and the configured :ref:`compatibility_disjoint_coefficient <compatibility-disjoint-coefficient-label>` for disjoint/excess genes.

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
      :return: Identifier/key to use.
      :rtype: int

.. todo::

  Add methods for the below.

.. py:module:: iznn
   :synopsis: Implements a spiking neural network (closer to in vivo neural networks) based on Izhikevich's 2003 model.

iznn
------

This module implements a spiking neural network. Neurons are based on the model described by::
    
  Izhikevich, E. M.
  Simple Model of Spiking Neurons
  IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 14, NO. 6, NOVEMBER 2003

See http://www.izhikevich.org/publications/spikes.pdf.

  .. inheritance-diagram:: iznn

  .. py:class:: IZNodeGene(BaseGene)

    Contains attributes for the iznn node genes and determines genomic distances.

  .. py:class:: IZGenome(DefaultGenome)

    Contains the parse_config class method for iznn genome configuration.

  .. py:class:: IZNeuron(bias, a, b, c, d, inputs)

    Sets up and simulates the iznn :term:`nodes <node>` (neurons).

    :param float bias: The bias of the neuron.
    :param float a: The time scale of the recovery variable.
    :param float b: The sensitivity of the recovery variable.
    :param float c: The after-spike reset value of the membrane potential.
    :param float d: The after-spike reset of the recovery variable.
    :param inputs: A list of (input key, weight) pairs for incoming connections.
    :type inputs: list(tuple(int, float))

    .. py:method:: advance(dt_msec)

      Advances simulation time for the neuron by the given time step in milliseconds. TODO: Currently has some numerical stability problems.

      :param float dt_msec: Time step in milliseconds.

    .. py:method:: reset()

      Resets all state variables.

  .. py:class:: IZNN(neurons, inputs, outputs)

    Sets up the network itself and simulates it using the connections and neurons.

    :param list neurons: The :py:class:`IZNeuron` instances needed.
    :param inputs: The :term:`input node` keys.
    :type inputs: list(int)
    :param outputs: The :term:`output node` keys.
    :type outputs: list(int)

    .. py:method:: set_inputs(inputs)

      Assigns input voltages.

      :param inputs: The input voltages for the :term:`input nodes <input node>`.
      :type inputs: list(float)

    .. py:method:: reset()

      Resets all neurons to their default state.

    .. py:method:: get_time_step_msec()

      Returns a suggested time step; currently hardwired to 0.05 - investigation of this (particularly effects on numerical stability issues) is planned.

      :return: Suggested time step in milliseconds.
      :rtype: float

    .. py:method:: advance(dt_msec)

      Advances simulation time for all neurons in the network by the input number of milliseconds.

      :param float dt_msec: How many milliseconds to advance the network.
      :return: The values for the :term:`output nodes <output node>`.
      :rtype: list(float)

    .. py:staticmethod:: create(genome, config)

      Receives a genome and returns its phenotype (a neural network).

      :param object genome: An IZGenome instance.
      :param object config: Configuration object.
      :return: An IZNN instance.
      :rtype: object

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

    A straightforward (no pun intended) :term:`feed-forward` neural network NEAT implementation.

    :param inputs: The input keys (IDs).
    :type inputs: list(int)
    :param outputs: The output keys.
    :type outputs: list(int)
    :param node_evals: A list of node descriptions, with each node represented by a list.
    :type node_evals: list(list(object))

    .. py:method:: activate(inputs)

      Feeds the inputs into the network and returns the resulting outputs.

      :param list inputs: The values for the :term:`input nodes <input node>`.
      :return: The values for the :term:`output nodes <output node>`.
      :rtype: list

    .. py:staticmethod:: create(genome, config)

      Receives a genome and returns its phenotype (a :py:class:`FeedForwardNetwork`).

.. py:module:: nn.recurrent
   :synopsis: A recurrent (but otherwise straightforward) neural network NEAT implementation.

nn.recurrent
----------------------

  .. py:class:: RecurrentNetwork(inputs, outputs, node_evals)

    A :term:`recurrent` (but otherwise straightforward) neural network NEAT implementation.

    :param inputs: The input keys (IDs).
    :type inputs: list(int)
    :param outputs: The output keys.
    :type outputs: list(int)
    :param node_evals: A list of node descriptions, with each node represented by a list.
    :type node_evals: list(list(object))

    .. py:method:: reset()

      Resets all node activations to 0 (necessary due to otherwise retaining state via recurrent connections).

    .. py:method:: activate(inputs)

      Feeds the inputs into the network and returns the resulting outputs.

      :param list inputs: The values for the :term:`input nodes <input node>`.
      :return: The values for the :term:`output nodes <output node>`.
      :rtype: list

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
      evaluations in :py:mod:`parallel`, etc.

      It is assumed that the fitness function does not modify the list of genomes,
      the genomes themselves (apart from updating the fitness member),
      or the configuration object.

      :param object fitness_function: The fitness function to use, with arguments specified above.
      :param n: The maximum number of generations to run (unlimited if ``None``).
      :type n: int or None
      :return: The best genome seen.
      :rtype: object

.. py:module:: reporting
   :synopsis: Makes possible reporter classes, which are triggered on particular events and may provide information to the user, may do something else such as checkpointing, or may do both.

reporting
-----------

  .. inheritance-diagram:: neat.reporting checkpoint.Checkpointer statistics.StatisticsReporter

  .. py:class:: ReporterSet

    Keeps track of the set of reporters and gives functions to dispatch them at appropriate points.

  .. py:class:: BaseReporter

    Definition of the reporter interface expected by ReporterSet. Inheriting from it will provide a set of ``dummy`` methods to be overridden as desired, as follows.

    .. py:method:: start_generation(generation)

      Called (by :py:meth:`Population.run`) at the start of each generation, prior to the invocation of the fitness function.

      :param int generation: The generation number.

    .. py:method:: end_generation(config, population, species)

      Called (by :py:meth:`Population.run`) at the end of each generation, after reproduction and speciation.

      :param object config: :py:class:`Config` configuration object.
      :param population: Current population, as a dict of unique genome ID/key vs genome.
      :type population: dict(int, object)
      :param object species: Current species set object, such as a :py:class:`DefaultSpeciesSet`.

    .. py:method:: post_evaluate(config, population, species, best_genome)

      Called (by :py:meth:`Population.run`) after the fitness function is finished.

      :param object config: :py:class:`Config` configuration object.
      :param population: Current population, as a dict of unique genome ID/key vs genome.
      :type population: dict(int, object)
      :param object species: Current species set object, such as a :py:class:`DefaultSpeciesSet`.
      :param object best_genome: The currently highest-fitness :term:`genome`. Ties are resolved pseudorandomly (by `dictionary <dict>` ordering).

    .. py:method:: post_reproduction(config, population, species)

      Not currently called, either by :py:meth:`Population.run` or by :py:class:`DefaultReproduction`. Note: New members of the population likely will not have a set species.

    .. py:method:: complete_extinction()

      Called (by :py:meth:`Population.run`) if complete extinction (due to stagnation) occurs, prior to
      (depending on the :ref:`reset_on_extinction <reset-on-extinction-label>` configuration setting)
      a new population being created or a :py:exc:`CompleteExtinctionException` being raised.

    .. py:method:: found_solution(config, generation, best)

      Called (by :py:meth:`Population.run`) prior to exiting if the configured :ref:`fitness threshold <fitness-threshold-label>` is met.
      (Note: Not called upon reaching the generation maximum and exiting for this reason.)

      :param object config: :py:class:`Config` configuration object.
      :param int generation: Generation number.
      :param object best: The currently highest-fitness :term:`genome`. Ties are resolved pseudorandomly (by `dictionary <dict>` ordering).

    .. py:method:: species_stagnant(sid, species)

      Called (by py:meth:`DefaultReproduction.reproduce`) for each species considered stagnant by the stagnation class (such as :py:class:`DefaultStagnation`).

      :param int sid: The species id/key.
      :param object species: The :py:class:`Species` object.

    .. py:method:: info(msg)

      Miscellaneous informational messages, from multiple parts of the library.

      :param str msg: Message to be handled.

  .. py:class:: StdOutReporter(show_species_detail)

    Uses print to output information about the run; an example reporter class.

    :param bool show_species_detail: Whether or not to show additional details about each species in the population.

.. todo::

  Add links to configuration file.

.. index:: fitness function

.. py:module:: reproduction
   :synopsis: Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents.

reproduction
-----------------

  .. py:class:: DefaultReproduction(config, reporters, stagnation)

    Handles creation of genomes, either from scratch or by sexual or asexual reproduction from parents. Implements the default NEAT-python reproduction
    scheme: explicit fitness sharing with fixed-time species stagnation. For class requirements, see :ref:`reproduction-interface-label`.

    :param dict config: Configuration object, in this implementation a dictionary.
    :param object reporters: A :py:class:`ReporterSet` object.
    :param object stagnation: A :py:class:`DefaultStagnation` object - current code partially depends on internals of this class (a TODO is noted to correct this)

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Provides defaults for ``elitism``, ``survival_threshold``, and ``min_species_size`` parameters and updates them from the configuration file.

      :param dict param_dict: Dictionary of parameters from configuration file.
      :return: Configuration object; considered opaque by rest of code, so current type returned is not required for interface.
      :rtype: dict

    .. py:classmethod:: write_config(f, param_dict)

      Required interface method. Saves ``elitism`` and ``survival_threshold`` (but not ``min_species_size``) parameters to new config file.

      :param file f: File object to write to.
      :param dict param_dict: Dictionary of current parameters in this implementation; more generally, reproduction config object.

    .. py:method:: create_new(genome_type, genome_config, num_genomes)

      Required interface method. Creates ``num_genomes`` new genomes of the given type using the given configuration. Also initializes ancestry information (empty tuple).

      :param class genome_type: Genome class (such as :py:class:`DefaultGenome` or :py:class:`IZGenome`) to create instances of.
      :param object genome_config: Opaque genome configuration object.
      :param int num_genomes: How many new genomes to create.
      :return: A dictionary (with the unique genome identifier as the key) of the genomes created.
      :rtype: dict(int, object)

    .. py:staticmethod:: compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size)

      Apportions desired number of members per species according to fitness (adjusted by :py:meth:`reproduce` to a 0-1 scale) from out of the desired population size.

      :param adjusted_fitness: Mean fitness for species members, adjusted to 0-1 scale (see below).
      :type adjusted_fitness: list(float)
      :param previous_sizes: Number of members of species in population prior to reproduction.
      :type previous_sizes: list(int)
      :param int pop_size: Desired population size, as input to :py:meth:`reproduce`.
      :param int min_species_size: Minimum number of members per species; can result in population size being above ``pop_size``.

    .. py:method:: reproduce(config, species, pop_size, generation)

      Required interface method. Creates the population to be used in the next generation from the given configuration instance, SpeciesSet instance, desired size of the
      population, and current generation number.  This method is called after all genomes have been evaluated and their ``fitness`` member assigned.  This method
      should use the stagnation instance given to the initializer to remove species deemed to have stagnated. Note: Determines relative fitnesses by transforming into
      (ideally) a 0-1 scale; however, if the top and bottom fitnesses are not at least 1 apart, the range may be less than 0-1, as a check against dividing by a too-small
      number. TODO: Make minimum difference configurable (defaulting to 1 to preserve compatibility).

      :param object config: A :py:class:`Config` instance.
      :param object species: A :py:class:`SpeciesSet` instance. As well as depending on some of the :py:class:`DefaultStagnation` internals, this method also depends on some of those of the ``SpeciesSet`` and its referenced species objects.
      :param int pop_size: Population size desired.
      :param int generation: Generation count.
      :return: New population, as a dict of unique genome ID/key vs genome.
      :rtype: dict(int, object)

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

.. index:: ! genomic distance

.. py:module:: species
   :synopsis: Divides the population into genome-based species.

species
-----------

  .. py:class:: Species(key, generation)

    Represents a :term:`species` and contains data about it such as members, fitness, and time stagnating (note: :py:class:`DefaultStagnation` manipulates many of these).

    :param int key: Identifier
    :param int generation: Initial generation of appearance

  .. py:class:: GenomeDistanceCache(config)

    Caches :term:`genomic distance` information to avoid repeated lookups (the :py:meth:`distance function <genome.DefaultGenome.distance>` is among the most
    time-consuming parts of the library, although most fitness functions are likely to far outweigh this). Called as a method with a pair of genomes to retrieve the distance.

  .. py:class:: DefaultSpeciesSet(config, reporters)

    Encapsulates the default speciation scheme by configuring it and performing the speciation function (placing genomes into species by genetic similarity).
    :py:class:`DefaultReproduction` currently depends on this having a ``species`` attribute consisting of a dictionary of species keys to species.

    :param object config: A configuration object (currently unused).
    :param object reporters: A :py:class:`ReporterSet` instance giving reporters to be notified about :term:`genomic distance` statistics.

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Currently, the only configuration parameter is the :ref:`compatibility_threshold <compatibility-threshold-label>`.

      :param param_dict: Dictionary of parameters from configuration file.
      :type param_dict: dict(str, str)
      :return: Configuration object; considered opaque by rest of code, so current type returned is not required for interface.
      :rtype: dict

    .. py:classmethod:: write_config(f, param_dict)

      Required interface method. Writes parameter(s) to new config file.

      :param file f: File object to write to.
      :param dict param_dict: Dictionary of current parameters in this implementation; more generally, stagnation config object.

    .. py:method:: speciate(config, population, generation)

      Required interface method. Place genomes into species by genetic similarity (:term:`genomic distance`). (The current code has a `docstring` stating that there may
      be a problem if all old species representatives are not dropped for each generation; it is not clear how this is consistent with the code
      in :py:meth:`DefaultReproduction.reproduce`, such as for ``elitism``.)

      :param object config: :py:class:`DefaultConfig` object.
      :param population: Population as per the output of :py:meth:`DefaultReproduction.reproduce`.
      :type population: dict(int, object)
      :param int generation: Current generation number.

    .. py:method:: get_species_id(individual_id)

      Required interface method (used by :py:class:`StdOutReporter`). Retrieves species id for a given genome id.

      :param int individual_id: Genome id/key.
      :return: Species id/key.
      :rtype: int

    .. py:method:: get_species(individual_id)

      Retrieves species object for a given genome id. May become a required interface method, and useful for some fitness functions already.

      :param int individual_id: Genome id/key.
      :return: :py:class:`Species` containing the genome corresponding to the id/key.
      :rtype: object

.. todo::

   Add more methods to the below for DefaultStagnation; try to figure out which ones are required interface methods; links re config file.

.. py:module:: stagnation
   :synopsis: Keeps track of whether species are making progress and removes ones that are not (for a configurable number of generations).

stagnation
--------------

  .. py:class:: DefaultStagnation(config, reporters)

    Keeps track of whether species are making progress and helps remove ones that, for a configurable number of generations, are not.

    :param object config: Configuration object; in this implementation, a `dictionary <dict>`, but should be treated as opaque outside this class.
    :param class reporters: A :py:class:`ReporterSet` with reporters that may need activating; not currently used.

    .. py:classmethod:: parse_config(param_dict)

      Required interface method. Provides defaults for ``species_fitness_func``, ``max_stagnation``, and ``species_elitism`` parameters and updates them from the
      configuration file.

      :param param_dict: Dictionary of parameters from configuration file.
      :type param_dict: dict(str, str)
      :return: Configuration object; considered opaque by rest of code, so current type returned is not required for interface.
      :rtype: dict

    .. py:classmethod:: write_config(f, param_dict)

      Required interface method. Saves parameters to new config file. **Has a default of 15 for species_elitism, but will be overridden by the default of 0 in parse_config.**

      :param file f: File object to write to.
      :param dict param_dict: Dictionary of current parameters in this implementation; more generally, stagnation config object.

.. Note: The notes below are not meant to be critical; I can see why the design decisions were made, for at least this iteration of the library.

.. py:module:: statistics
   :synopsis: Gathers and provides (to callers and/or to a file) information on genome and species fitness, which are the most-fit genomes, and similar.

statistics
-------------

.. note::
    * The most-fit genomes are based on the highest-fitness member of each generation; other genomes are not saved by this module, and it is assumed that fitnesses (as given by the :index:`fitness function <single: fitness function>`) are not relative to others in the generation (also assumed by the use of the :ref:`fitness threshold <fitness-threshold-label>` as a signal for exiting.
    * Currently keeps accumulating information in memory, which may be a problem in long runs.
    * Generally reports or records a per-generation list of values; the numeric position in the list may not correspond to the generation number if there has been a restart, such as via the :py:mod:`checkpoint` module.

  .. py:class:: StatisticsReporter(BaseReporter)

    Gathers (via the reporting interface) and provides (to callers and/or to a file) the most-fit genomes and information on genome and species fitness and species sizes. 

    .. py:method:: post_evaluate(config, population, species, best_genome)

      Called as part of the :py:class:`BaseReporter` interface after the evaluation at the start of each generation; see :py:meth:`BaseReporter.post_evaluate`.
      Information gathered includes a copy of the best genome in each generation and the fitnesses of each member of each species.

    .. py:method:: get_fitness_stat(f)

      Calls the given function on the genome fitness data from each recorded generation and returns the resulting list.

      :param function f: A function that takes a list of scores and returns a summary statistic (or, by returning a list or tuple, multiple statistics) such as ``mean`` or ``stdev``.
      :return: A list of the results from function f for each generation.
      :rtype: list

    .. py:method:: get_fitness_mean()

      Gets the per-generation average fitness. A wrapper for :py:meth:`get_fitness_stat` with the function being ``mean``.

      :return: List of mean genome fitnesses for each generation.
      :rtype: list(float)

    .. py:method:: get_fitness_stdev()

      Gets the per-generation standard deviation of the fitness. A wrapper for :py:meth:`get_fitness_stat` with the function being ``stdev``.

      :return: List of standard deviations of genome fitnesses for each generation.
      :rtype: list(float)

    .. py:method:: best_unique_genomes(n)

      Returns the ``n`` most-fit genomes, with no duplication (due to the most-fit genome passing unaltered to the next generation), sorted in decreasing fitness order.

      :param int n: Number of most-fit genomes to return.
      :return: List of ``n`` most-fit genomes (as genome objects).
      :rtype: list(object)

    .. py:method:: best_genomes(n)

      Returns the ``n`` most-fit genomes, possibly with duplicates, sorted in decreasing fitness order.

      :param int n: Number of most-fit genomes to return.
      :return: List of ``n`` most-fit genomes (as genome objects).
      :rtype: list(object)

    .. py:method:: best_genome()

      Returns the most-fit genome ever seen. A wrapper around :py:meth:`best_genomes`.

      :return: The most-fit genome.
      :rtype: object

    .. py:method:: get_species_sizes()

      Returns a by-generation list of lists of species sizes. Note that some values may be 0, if a species has either not yet been seen or has been removed due
      to :py:mod:`stagnation`; species without generational overlap may be more similar in :term:`genomic distance` than the configured
      :ref:`compatibility_threshold <compatibility-threshold-label>` would otherwise allow.

      :return: List of lists of species sizes.
      :rtype: list(list(int))

    .. py:method:: get_species_fitness(null_value='')

      Returns a by-generation list of lists of species fitnesses; the fitness of a species is determined by the ``mean`` fitness of the genomes in the species, as with
      the reproduction distribution by :py:class:`DefaultReproduction`. The ``null_value`` parameter is used for species not present in a particular generation (see above).

      :param str null_value: What to put in the list if the species is not present in a particular generation.
      :return: List of lists of species fitnesses.
      :rtype: list(list(float or str))

    .. py:method:: save_genome_fitness(delimiter=' ', filename='fitness_history.csv', with_cross_validation=False)

      Saves the population's best and mean fitness (using the `csv` package). At some point in the future, cross-validation fitness may be usable (via, for instance, the
      fitness function using alternative test situations/opponents and recording this in a ``cross_fitness`` attribute; this can be used for, e.g., preventing overfitting);
      currently, ``with_cross_validation`` should always be left at its ``False`` default.

      :param str delimiter: Delimiter between columns in the file; note that the default is not ',' as may be otherwise implied by the ``csv`` file extension (which refers to the package used).
      :param str filename: The filename to open (for writing, not appending) and write to.
      :param bool with_cross_validation: For future use; currently, leave at its ``False`` default.

    .. py:method:: save_species_count(delimiter=' ', filename='speciation.csv')

      Logs speciation throughout evolution, by tracking the number of genomes in each species. Uses :py:meth:`get_species_sizes`; see that method for more information.

      :param str delimiter: Delimiter between columns in the file; note that the default is not ',' as may be otherwise implied by the ``csv`` file extension (which refers to the package used).
      :param str filename: The filename to open (for writing, not appending) and write to.

    .. py:method:: save_species_fitness(delimiter=' ', null_value='NA', filename='species_fitness.csv')

      Logs species' mean fitness throughout evolution. Uses :py:meth:`get_species_fitness`; see that method for more information on, for instance, ``null_value``.

      :param str delimiter: Delimiter between columns in the file; note that the default is not ',' as may be otherwise implied by the ``csv`` file extension (which refers to the package used).
      :param str null_value: See :py:meth:`get_species_fitness`.
      :param str filename: The filename to open (for writing, not appending) and write to.

    .. py:method:: save()

      A wrapper for :py:meth:`save_genome_fitness`, :py:meth:`save_species_count`, and :py:meth:`save_species_fitness`; uses the default values for all three.
