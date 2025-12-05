
Installation
============

About The Examples
------------------

Because `neat-python` is still changing fairly rapidly, attempting to run examples with a significantly newer or older
version of the library will result in errors.  It is best to obtain matching example/library code by using one of the
two methods outlined below:

Install neat-python from PyPI using pip
---------------------------------------
To install the most recent release (version 1.1.0) from PyPI, you should run the command (as root or using `sudo`
as necessary)::

    pip install neat-python

Note that the examples are not included with the package installed from PyPI, so you should download the `source archive
for release 1.1.0
<https://github.com/CodeReclaimers/neat-python/releases/tag/v1.0.0>`_ and use the example code contained in it.

You may also just get the 1.1.0 release source, and install it directly (as shown below)
instead of `pip`.

Install neat-python from source
--------------------------------
Obtain the source code by either cloning the source repository::

    git clone https://github.com/CodeReclaimers/neat-python.git

or downloading the `source archive for release 1.1.0
<https://github.com/CodeReclaimers/neat-python/releases/tag/v1.1.0>`_.

Note that the most current code in the repository may not always be in the most polished state, but I do make sure the
tests pass and that most of the examples run.  If you encounter any problems, please open an `issue on GitHub
<https://github.com/CodeReclaimers/neat-python/issues>`_.

To install from source, run::

    pip install .

from the project root directory.

For development (editable install with dev dependencies)::

    pip install -e ".[dev]"

This installs the package in editable mode with testing tools (pytest, coverage, etc.).
