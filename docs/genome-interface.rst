.. _genome-interface-label:

Genome Interface
================

TODO: I will be coming back to this after I go through the module summaries; it is currently in a partially-revised state.

This is an outline of the minimal interface that is expected to be present on genome objects.

Class Methods
-------------

.. py:method:: parse_config(cls, param_dict)

  Takes a dictionary of configuration items, returns an object that will later be passed to the `write_config` method. This configuration object is considered to be opaque by the rest of the library.

.. py:method:: write_config(cls, f, config)

  Takes a file-like object and the configuration object created by parse_config. This method should write the configuration item definitions to the given file.

Initialization/Reproduction
------------------------------------

.. py:method:: __init__(self, key)

  Takes a unique genome instance identifier.  The initializer should create the following members:

        * `key`
        * `connections` - (gene_key, gene) pairs for the connection gene set.
        * `nodes` - (gene_key, gene) pairs for the node gene set.
        * `fitness`

.. py:method:: configure_new(self, config)

  Configure the genome as a new random genome based on the given configuration from the top-level `Config` object.

Crossover/Mutation
---------------------------

.. py:method:: configure_crossover(self, genome1, genome2, config)

  Configure the genome as a child of the given parent genomes.

.. py:method:: mutate(self, config)

  Apply mutation operations to the genome, using the given configuration.

.. index:: genomic distance

Speciation
---------------------

.. py:method:: distance(self, other, config)

  Returns the genomic distance between this genome and the other. This distance value is used to compute genome compatibility for speciation.

.. py:method:: size(self)

  Returns a measure of genome complexity. This object is currently only given to reporters at the end of a generation to indicate the complexity of the highest-fitness genome.  In the DefaultGenome class, this method currently returns (number of nodes, number of enabled connections).




