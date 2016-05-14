Welcome to NEAT-Python's documentation!
=======================================

**NOTE**: The examples are currently being moved into a separate repository (and
Python package) from the NEAT library itself.  During this time if you use the
code from the GitHub repository you may find that examples are missing and/or
not working.

NEAT (NeuroEvolution of Augmenting Topologies) is a method developed by Kenneth O. Stanley for evolving arbitrary neural
networks. NEAT-Python is a Python implementation of NEAT.

The core NEAT implementation is currently pure Python with no dependencies other than the Python standard library.
The `visualize` module requires graphviz, NumPy, and matplotlib, but it is not necessary to install these packages unless
you want to make use of these visualization utilities.  Some of the examples also make use of other libraries.

If you need an easy performance boost, JIT-enabled `PyPy
<http://pypy.org>`_ does a fantastic job, and may give you a ~10x speedup over CPython.  Once version 1.0 is out,
Built-in C and OpenCL network implementations will be added to speed up applications for which network evaluation is the primary bottleneck.

Support for HyperNEAT and other extensions to NEAT will also be added once the fundamental NEAT implementation is
more complete and stable.

**Please note:** the package and its usage may change significantly while it is still in alpha status.  Updating to
the most recent version is almost certainly going to break your code until the version number approaches 1.0.

For further information regarding general concepts and theory, please see `Selected Publications
<http://www.cs.ucf.edu/~kstanley/#publications>`_ on Stanley's website, or his recent `AMA on Reddit
<https://www.reddit.com/r/IAmA/comments/3xqcrk/im_ken_stanley_artificial_intelligence_professor>`_.

If you encounter any confusing or incorrect information in this documentation, please open an issue in the `GitHub project
<https://github.com/CodeReclaimers/neat-python>`_.


Contents:

.. toctree::
   :maxdepth: 2

   neat_overview
   installation
   config_file
   xor_example
   customization



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

