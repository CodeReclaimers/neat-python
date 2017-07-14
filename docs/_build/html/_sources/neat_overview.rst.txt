.. _neat-overview-label:

NEAT Overview
=============

:dfn:`NEAT` (NeuroEvolution of Augmenting Topologies) is an evolutionary algorithm that creates artificial neural networks. For a
detailed description of the algorithm, you should probably go read some of `Stanley's papers
<http://www.cs.ucf.edu/~kstanley/#publications>`_ on his website.

Even if you just want to get the gist of the algorithm, reading at least a couple of the early NEAT papers is a good
idea.  Most of them are pretty short, and do a good job of explaining concepts (or at least pointing
you to other references that will).  `The initial NEAT paper
<http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf>`_ is only 6 pages long, and Section II should be enough
if you just want a high-level overview.

In the current implementation of NEAT-Python, a population of individual :term:`genomes <genome>` is maintained.  Each genome contains
two sets of :term:`genes <gene>` that describe how to build an artificial neural network:

    1. :term:`Node` genes, each of which specifies a single neuron.
    2. :term:`Connection` genes, each of which specifies a single connection between neurons.

.. index:: ! fitness function

To evolve a solution to a problem, the user must provide a fitness function which computes a single real number
indicating the quality of an individual genome: better ability to solve the problem means a higher score.  The algorithm
progresses through a user-specified number of generations, with each generation being produced by reproduction (either
sexual or asexual) and mutation of the most fit individuals of the previous generation.

The reproduction and mutation operations may add nodes and/or connections to genomes, so as the algorithm proceeds
genomes (and the neural networks they produce) may become more and more complex.  When the preset number of generations
is reached, or when at least one individual (for a :ref:`fitness criterion function <fitness-criterion-label>` of ``max``; others are configurable)
exceeds the user-specified :ref:`fitness threshold <fitness-threshold-label>`, the algorithm terminates.

.. index:: ! crossover
.. index:: key
.. index:: ! homologous

One difficulty in this setup is with the implementation of :term:`crossover` - how does one do a crossover between two networks of differing structure?
NEAT handles this by keeping track of the origins of the nodes, with an :term:`identifying number <key>` (new, higher numbers are generated for each additional node). Those derived from a common ancestor (that are :term:`homologous`) are matched up for crossover, and connections are matched if the
nodes they connect have common ancestry. (There are variations in exactly how this is done depending on the implementation of NEAT; this paragraph describes how it is done in this implementation.)

.. index:: ! genomic distance
.. index:: ! disjoint
.. index:: ! excess

Another potential difficulty is that a structural mutation - as opposed to mutations in, for instance, the :term:`weights <weight>` of the connections - such as the addition of a node or connection can, while being promising for the future, be disruptive in the short-term (until it has been fine-tuned by less-disruptive
mutations). How NEAT deals with this is by dividing genomes into species, which have a close :term:`genomic distance` due to similarity, then having competition most intense within species, not between species (fitness sharing). How is genomic distance measured? It uses a combination of the number of non-homologous nodes and connections with measures of how much homologous nodes and connections have diverged since their common origin. (Non-homologous nodes and connections are termed :term:`disjoint` or :term:`excess`, depending on whether the :term:`numbers <key>` are from the same range or beyond that range; like most NEAT implementations, this one makes no distinction between the two.)
