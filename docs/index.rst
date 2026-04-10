Welcome to NEAT-Python's documentation!
=======================================

.. warning::
   **Breaking Changes — cumulative through v2.1**

   If you are upgrading from an earlier version, note the following breaking changes:

   * **v2.0**: ``CTRNN.create()`` no longer accepts a ``time_constant`` argument.
     Time constants are now per-node evolvable gene attributes, configured via
     ``time_constant_*`` parameters in the ``[DefaultGenome]`` config section.
     Checkpoints created with v1.x are not loadable in v2.0 or later.
   * **v1.0**: Innovation number tracking fully implemented per the NEAT paper;
     checkpoints from v0.x are not compatible. ``ThreadedEvaluator`` and
     ``DistributedEvaluator`` were removed — use ``ParallelEvaluator`` instead.
     All required configuration parameters must now be explicitly specified in
     the config file.

   See the :doc:`migration` guide for detailed upgrade instructions.

:abbr:`NEAT (NeuroEvolution of Augmenting Topologies)` is a method developed by Kenneth O. Stanley for evolving arbitrary neural
networks. NEAT-Python is a pure Python implementation of NEAT, with no dependencies other than the Python standard library.

Currently this library supports Python versions 3.8 through 3.14, as well as PyPy 3.

**For academic researchers:** See :doc:`academic_research` for guidance on using neat-python in research publications.

Many thanks to the original authors of this implementation, Cesar Gomes Miguel, Carolina Feher da Silva, and Marcio Lobo Netto!

.. note::
  Some of the example code has other dependencies. For your convenience there is a conda environment YAML file in the
  examples directory you can use to set up an environment that will support all of the current examples.
  TODO: Improve README.md file information for the examples.

For further information regarding general concepts and theory, please see `Selected Publications
<http://www.cs.ucf.edu/~kstanley/#publications>`_ on Stanley's website, or his `AMA on Reddit
<https://www.reddit.com/r/IAmA/comments/3xqcrk/im_ken_stanley_artificial_intelligence_professor>`_.

If you encounter any confusing or incorrect information in this documentation, please open an issue in the `GitHub project
<https://github.com/CodeReclaimers/neat-python>`_.

.. _toc-label:

Contents:

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   installation
   xor_example
   config_essentials

.. toctree::
   :maxdepth: 2
   :caption: Understanding NEAT

   neat_overview
   innovation_numbers
   glossary

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   academic_research
   config_file
   reproducibility
   cookbook
   customization
   activation
   ctrnn
   network_export

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   module_summaries
   genome-interface
   reproduction-interface

.. toctree::
   :maxdepth: 2
   :caption: Additional Resources

   faq
   troubleshooting
   migration

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

