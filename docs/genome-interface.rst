.. _genome-interface-label:

Genome Interface
================

.. index:: ! genome
.. index:: DefaultGenome

.. py:currentmodule:: genome

This is an outline of the minimal interface that is expected to be present on genome objects; example genome objects can be seen in :py:class:`DefaultGenome` and :py:class:`iznn.IZGenome`.

Class Methods
-------------

  :py:meth:`parse_config(cls, param_dict) <DefaultGenome.parse_config>`

  Takes a dictionary of configuration items, returns an object that will later be passed to the `write_config` method. This configuration object is considered to be opaque by the rest of the library.

  :py:meth:`write_config(cls, f, config) <DefaultGenome.write_config>`

  Takes a file-like object and the configuration object created by parse_config. This method should write the configuration item definitions to the given file.

Initialization/Reproduction
------------------------------------

  :py:class:`__init__(self, key) <DefaultGenome>`

  Takes a unique genome instance identifier.  The initializer should create the following members:

        * `key`
        * `connections` - (gene_key, gene) pairs for the connection gene set.
        * `nodes` - (gene_key, gene) pairs for the node gene set.
        * `fitness`

  :py:meth:`configure_new(self, config) <DefaultGenome.configure_new>`

  Configure the genome as a new random genome based on the given configuration from the top-level `Config` object.

Crossover/Mutation
---------------------------

  :py:meth:`configure_crossover(self, genome1, genome2, config) <DefaultGenome.configure_crossover>`

  Configure the genome as a child of the given parent genomes.

  :py:meth:`mutate(self, config) <DefaultGenome.mutate>`

  Apply mutation operations to the genome, using the given configuration.

.. index:: ! genomic distance

Speciation/Misc
------------------------

  :py:meth:`distance(self, other, config) <DefaultGenome.distance>`

  Returns the genomic distance between this genome and the other. This distance value is used to compute genome compatibility for speciation.

  :py:meth:`size(self) <DefaultGenome.size>`

  Returns a measure of genome complexity. This object is currently only given to reporters at the end of a generation to indicate the complexity of the highest-fitness genome.  In the DefaultGenome class, this method currently returns (number of nodes, number of enabled connections).




