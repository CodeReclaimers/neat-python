.. _activation-functions-label:

Overview of builtin activation functions
========================================

.. index:: ! activation function

Note that some of these :term:`functions <activation function>` are scaled differently from the canonical
versions you may be familiar with.  The intention of the scaling is to place
more of the functions' "interesting" behavior in the region :math:`\left[-1, 1\right] \times \left[-1, 1\right]`.

The implementation of these functions can be found in the :py:mod:`activations` module.

abs
---

.. figure:: activation-abs.png
   :scale: 50 %
   :alt: absolute value function


clamped
-------

.. figure:: activation-clamped.png
   :scale: 50 %
   :alt: clamped linear function

cube
----

.. figure:: activation-cube.png
   :scale: 50 %
   :alt: cubic function

exp
---

.. figure:: activation-exp.png
   :scale: 50 %
   :alt: exponential function


gauss
-----

.. figure:: activation-gauss.png
   :scale: 50 %
   :alt: gaussian function

hat
---

.. figure:: activation-hat.png
   :scale: 50 %
   :alt: hat function

.. _identity-label:

identity
--------

.. figure:: activation-identity.png
   :scale: 50 %
   :alt: identity function

inv
---

.. figure:: activation-inv.png
   :scale: 50 %
   :alt: inverse function

log
---

.. figure:: activation-log.png
   :scale: 50 %
   :alt: log function

relu
----

.. figure:: activation-relu.png
   :scale: 50 %
   :alt: rectified linear function

elu
----

.. figure:: activation-elu.png
   :scale: 50 %
   :alt: exponential rectified linear function

lelu
----

.. figure:: activation-lelu.png
   :scale: 50 %
   :alt: leaky rectified linear function

selu
----

.. figure:: activation-selu.png
   :scale: 50 %
   :alt: scaled exponential linear function

.. _sigmoid-label:

sigmoid
-------

.. figure:: activation-sigmoid.png
   :scale: 50 %
   :alt: sigmoid function

sin
---

.. figure:: activation-sin.png
   :scale: 50 %
   :alt: sine function

softplus
--------

.. figure:: activation-softplus.png
   :scale: 50 %
   :alt: soft-plus function

square
------

.. figure:: activation-square.png
   :scale: 50 %
   :alt: square function

tanh
----

.. figure:: activation-tanh.png
   :scale: 50 %
   :alt: hyperbolic tangent function
