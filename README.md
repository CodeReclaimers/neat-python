[![Build Status](https://travis-ci.org/CodeReclaimers/neat-python.svg)](https://travis-ci.org/CodeReclaimers/neat-python)
[![Coverage Status](https://coveralls.io/repos/CodeReclaimers/neat-python/badge.svg?branch=master&service=github)](https://coveralls.io/github/CodeReclaimers/neat-python?branch=master)

## FORK STATUS ##

This is a fork from [CodeReclaimers](https://github.com/CodeReclaimers/neat-python).
The main focus of this fork is to offer Non-dominated Sorting for Multiobjective Fitness. That means having more than one fitness value that should be optimized.
This should be done throught the implementation of [NSGA-II](https://ieeexplore.ieee.org/document/996017) as a Reproduction method.
Minor changes should be necessary on the main code to support list/tuple fitness values.

**NSGA-II is not currently implemented**

The current repository implements a hoverboard game/simulation to be used as a problem for testing the NSGA-II feature.
A CPTNN implementation with default reproduction is available as reference. Check the readme on `examples/nsga2` for more details.

![hoverboard-reference](https://i.imgur.com/SfPblbG.gif)

## About ##

NEAT (NeuroEvolution of Augmenting Topologies) is a method developed by Kenneth O. Stanley for evolving arbitrary neural
networks. This project is a pure-Python implementation of NEAT with no dependencies beyond the standard library. It was
forked from the excellent project by @MattKallada, and is in the process of being updated to provide more features and a
(hopefully) simpler and documented API.

For further information regarding general concepts and theory, please see
[Selected Publications](http://www.cs.ucf.edu/~kstanley/#publications) on Stanley's website.

`neat-python` is licensed under the [3-clause BSD license](https://opensource.org/licenses/BSD-3-Clause).

## Getting Started ##

If you want to try neat-python, please check out the repository, start playing with the examples (`examples/xor` is
a good place to start) and then try creating your own experiment.

The documentation, is available on [Read The Docs](http://neat-python.readthedocs.io).

## Citing ##

Here is a Bibtex entry you can use to cite this project in a publication. The listed authors are the maintainers of
all iterations of the project up to this point.

```
@misc{neat-python,
    Title = {neat-python},
    Author = {Alan McIntyre and Matt Kallada and Cesar G. Miguel and Carolina Feher da Silva},
    howpublished = {\url{https://github.com/CodeReclaimers/neat-python}}   
  }
```
