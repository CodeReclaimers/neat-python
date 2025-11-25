[![Tests](https://github.com/CodeReclaimers/neat-python/actions/workflows/tests.yml/badge.svg)](https://github.com/CodeReclaimers/neat-python/actions/workflows/tests.yml)
[![Docs](https://app.readthedocs.org/projects/neat-python/badge/?version=latest)](http://neat-python.readthedocs.io)
[![Coverage Status](https://coveralls.io/repos/CodeReclaimers/neat-python/badge.svg?branch=master&service=github)](https://coveralls.io/github/CodeReclaimers/neat-python?branch=master)
[![Downloads](https://static.pepy.tech/personalized-badge/neat-python?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/neat-python)

## About ##

NEAT (NeuroEvolution of Augmenting Topologies) is a method developed by Kenneth O. Stanley for evolving arbitrary neural
networks. This project is a pure-Python implementation of NEAT with no dependencies beyond the standard library. It was
forked from the excellent project by @MattKallada.

For further information regarding general concepts and theory, please see the [publications page](https://www.kenstanley.net/papers) of Stanley's current website.

`neat-python` is licensed under the [3-clause BSD license](https://opensource.org/licenses/BSD-3-Clause).  It is
currently only supported on Python 3.8 through 3.14, and pypy3.

## Features ##

* Pure Python implementation with no dependencies beyond the standard library
* Supports Python 3.8-3.14 and PyPy 3
* Reproducible evolution - Set random seeds for deterministic, repeatable experiments
* Parallel fitness evaluation using multiprocessing
* Network export to JSON format for interoperability
* Comprehensive documentation and examples

## Getting Started ##

If you want to try neat-python, please check out the repository, start playing with the examples (`examples/xor` is
a good place to start) and then try creating your own experiment.

The documentation is available on [Read The Docs](http://neat-python.readthedocs.io).

## Network Export ##

neat-python supports exporting trained networks to a JSON format that is framework-agnostic and human-readable. This allows you to:

- Convert networks to other formats (ONNX, TensorFlow, PyTorch, etc.) using third-party tools (the beginnings of a conversion system can be found in the `examples/export` directory)
- Inspect and debug network structure
- Share networks across platforms and languages
- Archive trained networks independently of neat-python

Example:
```python
import neat
from neat.export import export_network_json

# After training...
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

# Export to JSON
export_network_json(
    winner_net,
    filepath='my_network.json',
    metadata={'fitness': winner.fitness, 'generation': 42}
)
```

See [`docs/network-json-format.md`](docs/network-json-format.md) for complete format documentation and guidance for creating converters to other frameworks.

## Citing ##

Here are APA and Bibtex entries you can use to cite this project in a publication. The listed authors are the originators
and/or maintainers of all iterations of the project up to this point.  If you have contributed and would like your name added 
to the citation, please submit an issue.

APA
```
McIntyre, A., Kallada, M., Miguel, C. G., Feher de Silva, C., & Netto, M. L. neat-python [Computer software]
```

Bibtex
```
@software{McIntyre_neat-python,
author = {McIntyre, Alan and Kallada, Matt and Miguel, Cesar G. and Feher de Silva, Carolina and Netto, Marcio Lobo},
title = {{neat-python}}
}
```

## Thank you! ##
Many thanks to the folks who have [cited this repository](https://scholar.google.com/scholar?start=0&hl=en&as_sdt=5,34&sciodt=0,34&cites=15315010889003730796&scipsc=) in their own work. 
