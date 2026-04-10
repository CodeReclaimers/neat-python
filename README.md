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

## What's New in 2.1 ##

### Bug fixes

* **`fitness_criterion = min` now works correctly.** Previously, only the termination check
  honored this setting — best-genome tracking, stagnation detection, elite selection, crossover
  parent selection, spawn allocation, and statistics reporting all hardcoded "higher is better."
  All fitness comparisons throughout the library now respect the configured criterion.
* **Checkpoints no longer repeat work on restore.** Checkpoints are now saved after fitness
  evaluation (in `post_evaluate`) instead of after reproduction (in `end_generation`). Restoring
  a checkpoint skips the already-completed evaluation and proceeds directly to reproduction.
  For experiments with expensive fitness functions this eliminates potentially hours of redundant
  computation per restore. Checkpoint file `N` now means "generation N has been evaluated."
  Old checkpoint files (5-tuple format) are still loadable.
* **Reporter output no longer mixes generation boundaries.** The species detail table printed by
  `StdOutReporter` previously appeared in `end_generation` using the post-reproduction population,
  which belongs to the *next* generation. It now appears in `post_evaluate` alongside the fitness
  statistics, so all output under the "Running generation N" banner is consistent.
* **Fixed two double-buffer bugs in `CTRNN.advance`.** Incorrect buffer swapping could cause
  state corruption during multi-step CTRNN evaluation.
* **Fixed aggregation validation for builtins and callables.**

### NEAT paper compliance

Configurable options to more closely match Stanley & Miikkulainen (2002), with backward-compatible defaults:

* Connection gene matching by innovation number in the distance function (with separate
  `excess_coefficient`)
* Canonical fitness sharing (`fitness_sharing = canonical`)
* Proportional spawn allocation (`spawn_method = proportional`)
* Interspecies crossover (`interspecies_crossover_prob`)
* Dynamic compatibility threshold adjustment (`compatibility_threshold_adjustment`)
* 75% disable rule fix: replaces (rather than layers on) inherited enabled value
* Pruning of dangling nodes after deletion mutations
* Node gene distance contribution and enable/disable penalty are now configurable

### GPU acceleration

* Optional GPU-accelerated evaluation for CTRNN and Izhikevich networks via CuPy
  (`pip install 'neat-python[gpu]'`). Lazy imports — `import neat` never triggers a GPU dependency.
* CTRNN integration switched from forward Euler to exponential Euler (ETD1) for improved
  numerical stability.

### Other

* 55 new unit tests covering feature gaps (618 total).
* Sphinx 9.x documentation build compatibility fix.

## What's New in 2.0 ##

The CTRNN (Continuous-Time Recurrent Neural Network) implementation now supports **per-node evolvable time constants**. In v1.x, all nodes shared a single fixed time constant passed at network creation time. In v2.0, each node carries its own time constant as an evolved gene attribute, allowing the network to operate across multiple timescales simultaneously.

This is a breaking API change: `CTRNN.create(genome, config, time_constant)` is now `CTRNN.create(genome, config)`. Existing feedforward and discrete-time recurrent configurations require no changes.

For details on the change, its motivation, quantitative impact, and migration guide, see [CTRNN-CHANGES.pdf](examples/lorenz-ctrnn/docs/CTRNN-CHANGES.pdf).

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

You can also ask questions via the [experimental support agent](https://neat-python.recursive.support)!

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

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19024753.svg)](https://doi.org/10.5281/zenodo.19024753)

If you use this project in a publication, please cite both the software and the original NEAT paper. The listed authors are
the originators and/or maintainers of all iterations of the project up to this point. If you have contributed and would like
your name added to the citation, please submit an issue.

APA
```
McIntyre, A., Kallada, M., Miguel, C. G., Feher de Silva, C., & Netto, M. L. neat-python (Version 2.1.0) [Computer software]. https://doi.org/10.5281/zenodo.19024753
```

Bibtex
```
@software{McIntyre_neat-python,
author = {McIntyre, Alan and Kallada, Matt and Miguel, Cesar G. and Feher de Silva, Carolina and Netto, Marcio Lobo},
title = {{neat-python}},
version = {2.1.0},
doi = {10.5281/zenodo.19024753},
url = {https://github.com/CodeReclaimers/neat-python}
}
```

## Thank you! ##
Many thanks to the folks who have [cited this repository](https://scholar.google.com/scholar?start=0&hl=en&as_sdt=5,34&sciodt=0,34&cites=15315010889003730796&scipsc=) in their own work.

## About the Maintainer ##

neat-python is developed and maintained by Alan McIntyre
(CodeReclaimers LLC, ORCID: [0000-0002-8071-4219](https://orcid.org/0000-0002-8071-4219)).

Alan McIntyre is an independent consultant with 28+
years of software development experience and an MS in Applied Mathematics.
Specializations include computational geometry, CAD reverse engineering, C++ scientific computing, and Python scientific computing.

Available for research consulting and implementation engagements.
Full profile: [https://codereclaimers.com/consulting](https://codereclaimers.com/consulting)
Contact: consulting@codereclaimers.com

