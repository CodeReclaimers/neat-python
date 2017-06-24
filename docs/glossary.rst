Glossary
=========

.. I will check into the possibility of using sphinx.ext.imgmath instead of sphinx.ext.mathjax - I use lynx to check for browser compatibility
.. and HTML errors, and it does not do JavaScript, so I would greatly appreciate the substitution. - Allen (drallensmith)

.. glossary::
  :sorted:

  activation function
  aggregation function
  bias
  response
    These are the attributes of a :term:`node`. They determine the output of a node as follows:
    (TODO: Replace with proper mathematical formatting) activation_function(bias + (response*aggregation_function(input_connections)))
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
    :abbr:`NEAT (NeuroEvolution of Augmenting Topologies)` :doc:`algorithm <neat_overview>`, networks start with no hidden nodes, and evolve
    more complexity as necessary - thus "Augmenting Topologies".

  output node
    These are the :term:`nodes <node>` to which the network delivers outputs. They cannot be deleted (although connections to them can be) but
    can otherwise be altered normally.

  weight
  enabled
    These are the attributes of a :term:`connection`. If a connection is enabled, then the input to it (from a :term:`node`) is
    multiplied by the weight then sent to the output (to a node - possibly the same node, for a :term:`recurrent` neural network).
    If a connection is not enabled, then the output is 0; genes for such connections are the equivalent of `pseudogenes
    <http://pseudogene.org/background.php>`_ that, as in `in vivo <https://en.wikipedia.org/wiki/In_vivo>`_ evolution, can be reactivated at a later time.

  connection
    These connect between :term:`nodes <node>`, and give rise to the *network* in the term ``neural network``. Connections have two attributes,
    their :term:`weight` and whether or not they are :term:`enabled`; both are determined by their :term:`gene`.

  feedforward
    A neural network that is not :term:`recurrent` is feedforward - it has no loops. (Note that this means that it has no memory - no ability to take into account
    past events.) It can thus be described as a `DAG (Directed Acyclic Graph) <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_.

  recurrent
    A recurrent neural network has cycles in its topography. These may be a :term:`node` having a :term:`connection` back to itself, with (for a
    :term:`discrete time` neural network) the prior time period's output being provided to the node as one of its inputs. They may also have longer cycles,
    such as with output from node A going into node B (via a connection) and an output from node B going (via another connection) into node A. (This
    gives it a possibly-useful memory - an ability to take into account past events - unlike a :term:`feedforward` neural network; however, it also makes it
    harder to analyze.)

  continuous time
  discrete time
    A discrete time neural network proceeds in time steps, with processing at one :term:`node` followed by going through :term:`connections <connection>`
    to other nodes followed by processing at those other nodes. A continuous time neural network, such as the :doc:`ctrnn <ctrnn>` (continuous
    time :term:`recurrent` neural network) implemented in NEAT-Python, simulates a continuous process via differential equations (or other methods).

  genome
    Blah [include program documentation links...]

  genomic distance
    Blah [link!]

  gene
    Blah [link!]