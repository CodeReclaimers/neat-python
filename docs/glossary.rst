Glossary
=========

.. glossary::
  :sorted:

  attributes
    These are the properties of a :term:`node` (such as its :term:`activation function`) or :term:`connection` (such as whether it is :term:`enabled` or not)
    determined by its associated :term:`gene` (in the default implementation, in the :py:mod:`attributes` module).

  activation function
  aggregation function
  bias
  response
    These are the :term:`attributes` of a :term:`node`. They determine the output of a node as follows:
    :math:`\begin{equation}\operatorname{activation}(bias + (response * \operatorname{aggregation}(inputs)))\end{equation}`
    For available activation functions, see :ref:`activation-functions-label`

  node
    Also known as a neuron (as in a *neural* network). They are of three types:
    :term:`input <input node>`, :term:`hidden <hidden node>`, and :term:`output <output node>`. Nodes have one or more attributes, such
    as an :term:`activation function`; all are determined by their :term:`gene`.

  input node
    These are the :term:`nodes <node>` through which the network receives inputs. They cannot be deleted (although connections from them can be),
    cannot be the output end of a :term:`connection`, and have: no :term:`aggregation function`; a fixed :term:`bias` of 0; a fixed :term:`response` multiplier
    of 1; and a fixed :term:`activation function` of ``identity``.

  hidden node
    These are the :term:`nodes <node>` other than :term:`input nodes <input node>` and :term:`output nodes <output node>`. In the original
    NEAT (NeuroEvolution of Augmenting Topologies) :doc:`algorithm <neat_overview>`, networks start with no hidden nodes, and evolve
    more complexity as necessary - thus "Augmenting Topologies".

  homologous
    Descended from a common ancestor; two genes in NEAT from different genomes are either homologous or `:term:`disjoint`/excess.
    TODO: Explain further.

  disjoint
  excess
    These are genes in NEAT not descended from a common ancestor - i.e., not :term:`homologous` This implementation of NEAT, like most, does
    not distinguish between disjoint and excess genes. TODO: Explain further.

  output node
    These are the :term:`nodes <node>` to which the network delivers outputs. They cannot be deleted (although connections to them can be) but
    can otherwise be altered normally.

  weight
  enabled
    These are the :term:`attributes` of a :term:`connection`. If a connection is enabled, then the input to it (from a :term:`node`) is
    multiplied by the weight then sent to the output (to a node - possibly the same node, for a :term:`recurrent` neural network).
    If a connection is not enabled, then the output is 0; genes for such connections are the equivalent of `pseudogenes
    <http://pseudogene.org/background.php>`_ that, as in `in vivo <https://en.wikipedia.org/wiki/In_vivo>`_ evolution, can be reactivated at a later time.
    (Some versions of NEAT give a chance, such as 25%, that a disabled connection will be enabled during :term:`crossover`; in the future, this may be
    implemented as an option in this implementation.)

  connection
    These connect between :term:`nodes <node>`, and give rise to the *network* in the term ``neural network``. For non-loopback (directly :term:`recurrent`)
    connections, they are equivalent to biological synapses. Connections have two attributes, their :term:`weight` and whether or not they are
    :term:`enabled`; both are determined by their :term:`gene`.

  feedforward
  feed-forward
    A neural network that is not :term:`recurrent` is feedforward - it has no loops. (Note that this means that it has no memory - no ability to take into account
    past events.) It can thus be described as a `DAG (Directed Acyclic Graph) <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_.

  recurrent
    A recurrent neural network has cycles in its topography. These may be a :term:`node` having a :term:`connection` back to itself, with (for a
    :term:`discrete-time` neural network) the prior time period's output being provided to the node as one of its inputs. They may also have longer cycles,
    such as with output from node A going into node B (via a connection) and an output from node B going (via another connection) into node A. (This
    gives it a possibly-useful memory - an ability to take into account past events - unlike a :term:`feedforward` neural network; however, it also makes it
    harder to work with in some respects.)

  continuous-time
  discrete-time
    A discrete-time neural network (which should be assumed unless specified otherwise) proceeds in time steps, with processing at one :term:`node`
    followed by going through :term:`connections <connection>` to other nodes followed by processing at those other nodes, eventually giving the output.
    A continuous-time neural network, such as the :doc:`ctrnn <ctrnn>` (continuous-time :term:`recurrent` neural network) implemented in NEAT-Python,
    simulates a continuous process via differential equations (or other methods).

  genome
    The set of :term:`genes <gene>` that together code for a (neural network) phenotype. TODO: Add appropriate links.

  genomic distance
    An approximate measure of the difference between :term:`genomes <genome>`, used in dividing the population into :term:`species`. TODO: Explain
    further, put in links, etc...

  genetic distance
    The distance between two :term:`homologous` :term:`genes <gene>`, added up as part of the :term:`genomic distance`. Also sometimes used as
    a synonym for :term:`genomic distance`.

  gene
    The information coding (in the current implementation) for a particular aspect (:term:`node` or :term:`connection`) of a neural network phenotype.
    TODO: Add links.

  species
    Subdivisions of the population into groups of similar (by the :term:`genomic distance` measure) individuals (:term:`genomes <genome>`),
    which compete among themselves but share fitness relative to the rest of the population. This is, among other things, a mechanism to try to avoid the
    quick elimination of high-potential topological mutants that have an initial poor fitness prior to smaller "tuning" changes. TODO: Add links.

  crossover
    The process in sexual reproduction in which two :term:`genomes <genome>` are combined. This involves the combination of :term:`homologous`
    genes and the copying (from the highest-fitness genome) of :term:`disjoint/excess <disjoint>` genes. TODO: Add links.

:ref:`Table of Contents <toc-label>`