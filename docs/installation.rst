
Installation
============

About The Examples
------------------

Much of the sample code uses libraries that are not required to use `neat-python` itself.  Below are all the
packages you should install in addition to `neat-python` if you want to be able to run all the included examples.  **None
of these libraries are required if you are using `neat-python` in your own code and you do not import `neat.visualize`.**

`graphviz
<https://pypi.python.org/pypi/graphviz>`_

`maptlotlib
<http://matplotlib.org/users/installing.html>`_

`gizeh
<https://pypi.python.org/pypi/gizeh>`_

`moviepy
<https://pypi.python.org/pypi/moviepy>`_

`Pillow
<https://pypi.python.org/pypi/Pillow>`_


Because `neat-python` is still changing fairly rapidly, attempting to run examples with a significantly newer or older
version of the library will result in errors.  It is best to obtain matching example/library code by using one of the
two methods outlined below:

Install neat-python from PyPI using pip
---------------------------------------
To install the most recent release (version 0.7) from PyPI, you should run the command (as root or using `sudo`
as necessary)::

    pip install neat-python

Note that the examples are not included with the package installed from PyPI, so you should download the `source archive
for release 0.7
<https://github.com/CodeReclaimers/neat-python/releases/tag/v0.7>`_ and use the example code contained in it.

You may also just download the 0.7 release source, and install it directly using `setup.py` (as shown below)
instead of `pip`.

Install neat-python from source using setup.py
----------------------------------------------
Obtain the source code by either cloning the source repository::

    git clone https://github.com/CodeReclaimers/neat-python.git

or downloading the `source archive
for release 0.7
<https://github.com/CodeReclaimers/neat-python/releases/tag/v0.7>`_.

Note that the most current code in the repository may not always be in the most polished state, but I do make sure the
tests pass and that most of the examples run.  If you encounter any problems, please open an `issue on GitHub
<https://github.com/CodeReclaimers/neat-python/issues>`_.

To install from source, simply run::

    python setup.py install

from the directory containing setup.py.
