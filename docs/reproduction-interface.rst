.. _reproduction-interface-label:

Reproduction Interface
======================

This is an outline of the minimal interface that is expected to be present on reproduction objects.  Each Population
instance will create exactly one instance of the reproduction class in `Population.__init__` regardless of the
configuration or arguments provided to `Population.__init__`.

Class Methods
-------------

`parse_config(cls, param_dict)` - Takes a dictionary of configuration items, returns an object that will later
be passed to the `write_config` method.  This configuration object is considered to be opaque by the rest
of the library.

`write_config(cls, f, config)` - Takes a file-like object and the configuration object created by parse_config.
This method should write the configuration item definitions to the given file.

Initialization
--------------

`__init__(self, config, reporters, stagnation)` - Takes the reproduction configuration sub-object
(as returned by this class's ``parse_config`` method — for ``DefaultReproduction`` this is a
``DefaultClassConfig`` with fields like ``elitism``, ``survival_threshold``, ``min_species_size``,
etc.), a ``ReporterSet`` instance, and a stagnation object instance.

.. warning::
   The ``config`` parameter passed to ``__init__`` is the **reproduction sub-config**, not the
   top-level ``Config`` object. Note that the ``config`` argument passed to
   ``reproduce(self, config, species, pop_size, generation)`` IS the top-level ``Config`` —
   the same parameter name refers to different objects in ``__init__`` and ``reproduce``.

Other methods
-------------

`create_new(self, genome_type, genome_config, num_genomes):` - Create `num_genomes` new genomes of the given type
using the given configuration.

`reproduce(self, config, species, pop_size, generation):` - Creates the population to be used in the next generation
from the given configuration instance, SpeciesSet instance, desired size of the population, and current generation
number.  This method is called after all genomes have been evaluated and their `fitness` member assigned.  This method
should use the stagnation instance given to the initializer to remove species it deems to have stagnated.
