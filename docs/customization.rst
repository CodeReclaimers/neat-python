.. _customization-label:

Customizing Behavior
====================

.. default-role:: any

NEAT-Python allows the user to provide drop-in replacements for some parts of the NEAT algorithm, which hopefully
makes it easier to implement common variations of the algorithm as mentioned in the literature.  If
you find that you'd like to be able to customize something not shown here, please submit an issue on GitHub.

.. index:: activation function

New activation functions
------------------------
New :term:`activation functions <activation function>` are registered with your :py:class:`Config <config.Config>` instance, prior to creation of the
:py:class:`Population <population.Population>` instance, as follows::

    def sinc(x):
        return 1.0 if x == 0 else sin(x) / x

    config.genome_config.add_activation('my_sinc_function', sinc)

The first argument to :py:meth:`add_activation <genome.DefaultGenomeConfig.add_activation>` is the name by which this activation function will be referred to in the configuration settings file.

This is demonstrated in the `memory-fixed
<https://github.com/CodeReclaimers/neat-python/tree/master/examples/memory-fixed>`_ example.

.. note::

  This method is only valid when using the :py:class:`DefaultGenome <genome.DefaultGenome>` implementation, with the method being found in
  the :py:class:`DefaultGenomeConfig <genome.DefaultGenomeConfig>` implementation; different genome implementations
  may require a different method of registration.

.. index:: reporting

Reporting/logging
-----------------

The Population class makes calls to a collection of zero or more reporters at fixed points during the evolution
process.  The user can add a custom reporter to this collection by calling Population.add_reporter and providing
it with an object which implements the same interface as `BaseReporter` (in :py:mod:`reporting.py <reporting>`), probably partially by subclassing it.

:py:class:`StdOutReporter <reporting.StdOutReporter>`, :py:class:`StatisticsReporter <statistics.StatisticsReporter>`, and :py:class:`Checkpointer <checkpoint.Checkpointer>` may be useful as examples of the behavior you can add using a reporter.

.. index:: genome
.. index:: DefaultGenome

New genome types
----------------

To use a different genome type, you can create a custom class whose interface matches that of
`DefaultGenome` and pass this as the ``genome_type`` argument to the `Config` constructor. The minimum genome type interface is documented here: :ref:`genome-interface-label`.

This is demonstrated in the `circuit evolution
<https://github.com/CodeReclaimers/neat-python/blob/master/examples/circuits/evolve.py>`_ example.

Alternatively, you can subclass `DefaultGenome` in cases where you need to just add some extra behavior.
This is done in the `OpenAI lander
<https://github.com/CodeReclaimers/neat-python/blob/master/examples/openai-lander/evolve.py>`_ example to
add an evolvable per-genome reward discount value. It is also done in the :py:mod:`iznn` setup, with :py:class:`IZGenome <iznn.IZGenome>`.

.. index:: species

Speciation scheme
-----------------

To use a different speciation scheme, you can create a custom class whose interface matches that of
:py:class:`DefaultSpeciesSet <species.DefaultSpeciesSet>` and pass this as the ``species_set_type`` argument to the `Config` constructor.

.. note::

  TODO: Further document species set interface (some done in module_summaries)

.. note::

  TODO: Include example

.. index:: stagnation

Species stagnation scheme
-------------------------

The default species stagnation scheme is a simple fixed stagnation limit--when a species exhibits
no improvement for a fixed number of generations, all its members are removed from the simulation. This
behavior is encapsulated in the :py:class:`DefaultStagnation class <stagnation.DefaultStagnation>`.

To use a different species stagnation scheme, you must create a custom class whose interface matches that
of `DefaultStagnation`, and provide it as the ``stagnation_type`` argument to the `Config` constructor.

This is demonstrated in the `interactive 2D image
<https://github.com/CodeReclaimers/neat-python/blob/master/examples/picture2d/evolve_interactive.py>`_ example.

.. index:: reproduction
.. index:: DefaultReproduction

Reproduction scheme
-------------------

The default reproduction scheme uses explicit fitness sharing.  This behavior is encapsulated in the
`DefaultReproduction` class.  The minimum reproduction type interface is documented here: :ref:`reproduction-interface-label`

To use a different reproduction scheme, you must create a custom class whose interface matches that
of `DefaultReproduction`, and provide it as the ``reproduction_type`` argument to the `Config` constructor.

.. note:: 

  TODO: Include example
