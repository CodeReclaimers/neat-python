.. _genome-interface-label:

Genome Interface
================

This is an outline of the minimal interface that is expected to be present on genome objects.

Class Methods
-------------

`parse_config(cls, param_dict)` - Takes a dictionary of configuration items, returns an object that will later
be passed to the `write_config` method.  This configuration object is considered to be opaque by the rest
of the library.

`write_config(cls, f, config)` - Takes a file-like object and the configuration object created by parse_config.
This method should write the configuration item definitions to the given file.

`create(cls, config, key)` - Create and return a new genome instance with the given key, using the given configuration
object.

Initializer
-----------

`__init__(self, key, config)` - Takes a unique genome instance identifier and the top-level Config object.  The
initializer should create the following members:

        * `key`
        * `connections` - (gene_key, gene) pairs for the connection gene set.
        * `nodes` - (gene_key, gene) pairs for the node gene set.
        * `fitness`
        * `cross_fitness`


Mutation/reproduction
---------------------

`mutate(self, config)` - Apply mutation operations to the genome, using the given configuration.

`crossover(self, other, key, config)` - Crosses over parents' genomes and returns a child.

`distance(self, other, config)` - Returns the genetic distance between this genome and the other. This distance value
is used to compute genome compatibility for speciation.

`size(self)` - Returns a measure of genome complexity. This object is currently only given to reporters at the
end of a generation to indicate the complexity of the highest-fitness genome.  In the DefaultGenome class,
this method currently returns (number of nodes, number of enabled connections).




