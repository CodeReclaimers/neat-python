
Customizing Behavior
====================

NEAT-Python allows the user to provide drop-in replacements for some parts of the NEAT algorithm, and attempts
to allow easily implementing common variations of the algorithm mentioned in the literature.  If
you find that you'd like to be able to customize something not shown here, please submit an issue on GitHub.

Adding new activation functions
-------------------------------
To register a new activation function, you only need to call `neat.activation_functions.add` with your new
function and the name by which you want to refer to it in the configuration file::

    def sinc(x):
        return 1.0 if x == 0 else sin(x) / x

    neat.activation_functions.add('my_sinc_function', sinc)

Note that you must register the new function before you load the configuration file.

This is demonstrated in the `memory
<https://github.com/CodeReclaimers/neat-python/tree/master/examples/memory>`_ example.

Reporting/logging
-----------------

The Population class makes calls to a collection of zero or more reporters at fixed points during the evolution
process.  The user can add a custom reporter to this collection by calling Population.add_reporter and providing
it with an object which implements the same interface as `BaseReporter`.

`StdOutReporter` and `StatisticsReporter` in `reporting.py
<https://github.com/CodeReclaimers/neat-python/blob/master/neat/reporting.py#L56>`_ may be useful as examples of the
behavior you can add using a reporter.

Species stagnation scheme
-------------------------

To use a different species stagnation scheme, you must create and register a custom class whose interface matches that
of `DefaultStagnation` and set the `stagnation_type` of your Config instance to this class.

This is demonstrated in the `interactive 2D image
<https://github.com/CodeReclaimers/neat-python/blob/master/examples/picture2d/interactive.py>`_ example.

TODO: document stagnation interface

Reproduction scheme
-------------------

The default reproduction scheme uses explicit fitness sharing and a fixed species stagnation limit.  This behavior
is encapsulated in the DefaultReproduction class.

TODO: document reproduction interface, include example

EXAMPLE: ???

Speciation
----------

If you need to change the speciation scheme, you should subclass `Population` and override the `_speciate` method (or,
if you must, `monkey patch/duck punch
<https://en.wikipedia.org/wiki/Monkey_patch>`_ it).

EXAMPLE: ???

Diversity
---------

To use a different diversity scheme, you can create a custom class whose interface matches that of
`ExplicitFitnessSharing` and set the `diversity_type` of your Config instance to this class.

TODO: document, include example

EXAMPLE: ???

Using different genome types
----------------------------

To use a different genome type, you can create a custom class whose interface matches that of
`Genome` or `FFGenome`, and set the `genotype` member of your Config instance to this class.

TODO: document genome interface, include example

EXAMPLE: PySpice circuit evolution

Using a different gene type
---------------------------

To use a different gene type, you can create a custom class whose interface matches that of
`NodeGene` or `ConnectionGene`, and set the `node_gene_type` or `conn_gene_type` member,
respectively, of your Config instance to this class.

TODO: document, include example

EXAMPLE: ???  reference the IZNN genes?

Using a different network type
------------------------------

EXAMPLE: ???

Cross-validation
----------------

EXAMPLE: ???

Custom evaluators
-----------------

TODO: Include a couple of evaluators to take the place of a simple function: one that uses
a thread pool, and another that farms out evaluation to clients that connect over the network.
