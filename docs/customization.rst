
Customizing Behavior
====================

NEAT-Python allows the user to provide drop-in replacements for some parts of the NEAT algorithm, which hopefully
makes it easier to implement common variations of the algorithm as mentioned in the literature.  If
you find that you'd like to be able to customize something not shown here, please submit an issue on GitHub.

New activation functions
------------------------
New activation functions are registered with your `Config` instance, prior to creation of the `Population` instance,
as follows::

    def sinc(x):
        return 1.0 if x == 0 else sin(x) / x

    config.genome_config.add_activation('my_sinc_function', sinc)

The first argument to `add_activation` is the name by which this activation function will be referred to in the
configuration settings file.

This is demonstrated in the `memory-fixed
<https://github.com/CodeReclaimers/neat-python/tree/master/examples/memory-fixed>`_ example.

NOTE: This method is only valid when using the `DefaultGenome` implementation--different genome implementations
may require a different method of registration.

Reporting/logging
-----------------

The Population class makes calls to a collection of zero or more reporters at fixed points during the evolution
process.  The user can add a custom reporter to this collection by calling Population.add_reporter and providing
it with an object which implements the same interface as `BaseReporter`.

`StdOutReporter` and `StatisticsReporter` in `reporting.py
<https://github.com/CodeReclaimers/neat-python/blob/master/neat/reporting.py#L56>`_ may be useful as examples of the
behavior you can add using a reporter.

TODO: document reporter interface

New genome types
----------------

To use a different genome type, you can create a custom class whose interface matches that of
`DefaultGenome` and pass this as the `genome_type` argument to the `Config` constructor.

This is demonstrated in the `circuit evolution
<https://github.com/CodeReclaimers/neat-python/blob/master/examples/circuits/evolve.py>`_ example.

TODO: document genome interface

Speciation scheme
-----------------

To use a different speciation scheme, you can create a custom class whose interface matches that of
`DefaultSpeciesSet` and pass this as the `species_set_type` argument to the `Config` constructor.

TODO: document species set interface

TODO: include example

Species stagnation scheme
-------------------------

The default species stagnation scheme is a simple fixed stagnation limit--when a species exhibits
no improvement for a fixed number of generations, all its members are removed from the simulation. This
behavior is encapsulated in the `DefaultStagnation` class.

To use a different species stagnation scheme, you must create a custom class whose interface matches that
of `DefaultStagnation`, and provide it as the `stagnation_type` argument to the `Config` constructor.

This is demonstrated in the `interactive 2D image
<https://github.com/CodeReclaimers/neat-python/blob/master/examples/picture2d/interactive.py>`_ example.

TODO: document stagnation interface

Reproduction scheme
-------------------

The default reproduction scheme uses explicit fitness sharing.  This behavior is encapsulated in the
`DefaultReproduction` class.

To use a different reproduction scheme, you must create a custom class whose interface matches that
of `DefaultReproduction`, and provide it as the `reproduction_type` argument to the `Config` constructor.

TODO: document reproduction interface

TODO: include example
