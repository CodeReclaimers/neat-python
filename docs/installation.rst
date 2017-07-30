
Installation
============

About The Examples
------------------

Because `neat-python` is still changing fairly rapidly, attempting to run examples with a significantly newer or older
version of the library will result in errors.  It is best to obtain matching example/library code by using one of the
two methods outlined below:

Install neat-python from PyPI using pip
---------------------------------------
To install the most recent release (version 0.92) from PyPI, you should run the command (as root or using `sudo`
as necessary)::

    pip install neat-python

Note that the examples are not included with the package installed from PyPI, so you should download the `source archive
for release 0.92
<https://github.com/CodeReclaimers/neat-python/releases/tag/v0.92>`_ and use the example code contained in it.

You may also just get the 0.92 release source, and install it directly using `setup.py` (as shown below)
instead of `pip`.

Install neat-python from source using setup.py
----------------------------------------------
Obtain the source code by either cloning the source repository::

    git clone https://github.com/CodeReclaimers/neat-python.git

or downloading the `source archive for release 0.92
<https://github.com/CodeReclaimers/neat-python/releases/tag/v0.92>`_.

Note that the most current code in the repository may not always be in the most polished state, but I do make sure the
tests pass and that most of the examples run.  If you encounter any problems, please open an `issue on GitHub
<https://github.com/CodeReclaimers/neat-python/issues>`_.

To install from source, simply run::

    python setup.py install

from the directory containing setup.py.
