
Overview of the basic XOR example (xor2.py)
===========================================

.. default-role:: any

The xor2.py example, shown in its entirety at the bottom of this page, evolves a network that implements the two-input
XOR function:

=======  =======  ======
Input 1  Input 2  Output
=======  =======  ======
0        0        0
0        1        1
1        0        1
1        1        0
=======  =======  ======

.. index:: ! fitness function

Fitness function
----------------

The key thing you need to figure out for a given problem is how to measure the fitness of the :term:`genomes <genome>` that are produced
by NEAT.  Fitness is expected to be a Python `float` value.  If genome A solves your problem more successfully than genome B,
then the fitness value of A should be greater than the value of B.  The absolute magnitude and signs of these fitnesses
are not important, only their relative values.

In this example, we create a :term:`feed-forward` neural network based on the genome, and then for each case in the
table above, we provide that network with the inputs, and compute the network's output.  The error for each genome
is :math:`1 - \sum_i (e_i - a_i)^2` between the expected (:math:`e_i`) and actual (:math:`a_i`) outputs, so that if the
network produces exactly the expected output, its fitness is 1, otherwise it is a value less than 1, with the fitness
value decreasing the more incorrect the network responses are.

This fitness computation is implemented in the ``eval_genomes`` function.  This function takes two arguments: a list
of genomes (the current population) and the active configuration.  neat-python expects the fitness function to calculate
a fitness for each genome and assign this value to the genome's ``fitness`` member.

Running NEAT
------------

Once you have implemented a fitness function, you mostly just need some additional boilerplate code that carries out
the following steps:

* Create a :py:class:`neat.config.Config <config.Config>` object from the configuration file (described in the :doc:`config_file`).

* Create a :py:class:`neat.population.Population <population.Population>` object using the ``Config`` object created above.

* Call the :py:meth:`run <population.Population.run>` method on the ``Population`` object, giving it your fitness function and (optionally) the maximum number of generations you want NEAT to run.

After these three things are completed, NEAT will run until either you reach the specified number of generations, or
at least one genome achieves the :ref:`fitness_threshold <fitness-threshold-label>` value you specified in your config file.

Getting the results
-------------------

Once the call to the population object's ``run`` method has returned, you can query the ``statistics`` member of the
population (a :py:class:`neat.statistics.StatisticsReporter <statistics.StatisticsReporter>` object) to get the best genome(s) seen during the run.
In this example, we take the 'winner' genome to be that returned by ``pop.statistics.best_genome()``.

Other information available from the default statistics object includes per-generation mean fitness, per-generation standard deviation of fitness,
and the best N genomes (with or without duplicates).

Visualizations
--------------

Functions are available in the `visualize module
<https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/visualize.py>`_ to plot the best
and average fitness vs. generation, plot the change in species vs. generation, and to show the structure
of a network described by a genome.

Example Source
--------------

NOTE: This page shows the source and configuration file for the current version of neat-python available on
GitHub.  If you are using the version 0.92 installed from PyPI, make sure you get the script and config file from
the `archived source for that release <https://github.com/CodeReclaimers/neat-python/releases/tag/v0.92>`_.

Here's the entire example:

.. literalinclude:: ../examples/xor/evolve-feedforward.py

and here is the associated config file:

.. literalinclude:: ../examples/xor/config-feedforward