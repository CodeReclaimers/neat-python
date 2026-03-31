
Installation
============

About The Examples
------------------

Because `neat-python` is still changing fairly rapidly, attempting to run examples with a significantly newer or older
version of the library will result in errors.  It is best to obtain matching example/library code by using one of the
two methods outlined below:

Install neat-python from PyPI using pip
---------------------------------------
To install the most recent release from PyPI, you should run the command (as root or using `sudo`
as necessary)::

    pip install neat-python

Note that the examples are not included with the package installed from PyPI, so you should download the `source archive
<https://github.com/CodeReclaimers/neat-python/releases>`_ and use the example code contained in it.

Install neat-python from source
--------------------------------
Obtain the source code by either cloning the source repository::

    git clone https://github.com/CodeReclaimers/neat-python.git

or downloading the latest `source archive
<https://github.com/CodeReclaimers/neat-python/releases>`_.

Note that the most current code in the repository may not always be in the most polished state, but I do make sure the
tests pass and that most of the examples run.  If you encounter any problems, please open an `issue on GitHub
<https://github.com/CodeReclaimers/neat-python/issues>`_.

To install from source, run::

    pip install .

from the project root directory.

For development (editable install with dev dependencies)::

    pip install -e ".[dev]"

This installs the package in editable mode with testing tools (pytest, coverage, etc.).

Optional extras
---------------

neat-python supports optional dependency groups that can be installed via pip extras::

    pip install 'neat-python[gpu]'       # GPU acceleration (CuPy, requires NVIDIA GPU)
    pip install 'neat-python[examples]'  # Dependencies for running examples
    pip install 'neat-python[docs]'      # Documentation building tools
    pip install 'neat-python[all]'       # Everything (includes GPU)

The ``[gpu]`` extra installs CuPy for GPU-accelerated CTRNN and Izhikevich network evaluation.
See :doc:`ctrnn` for details.
