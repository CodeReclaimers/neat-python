NEAT Overview
=============

:dfn:`NEAT` (NeuroEvolution of Augmenting Topologies) is an evolutionary algorithm that creates artificial networks. For a
detailed description of the algorithm, you should probably go read some of `Stanley's papers
<http://www.cs.ucf.edu/~kstanley/#publications>`_ on his website.

Even if you just want to get the gist of the algorithm, reading at least a couple of the early NEAT papers is a good
idea.  Most of them are pretty short, and do a good job of explaining concepts (or at least pointing
you to other references that will).  `The initial NEAT paper
<http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf>`_ is only 6 pages long, and Section II should be enough
if you just want a high-level overview.


In the current implementation of NEAT-Python, a population of individual genomes is maintained.  Each genome contains
two sets of genes that describe how to build an artificial neural network:

    1. :term:`Node` genes, each of which specifies a single neuron.
    2. :term:`Connection` genes, each of which specifies a single connection between neurons.

To evolve a solution to a problem, the user must provide a fitness function which computes a single real number
indicating the quality of an individual genome: better ability to solve the problem means a higher score.  The algorithm
progresses through a user-specified number of generations, with each generation being produced by reproduction (either
sexual or asexual) and mutation of the most fit individuals of the previous generation.

The reproduction and mutation operations may add nodes and/or connections to genomes, so as the algorithm proceeds
genomes (and the neural networks they produce) may become more and more complex.  When the preset number of generations
is reached, or when at least one individual exceeds the user-specified fitness threshold, the algorithm terminates.

