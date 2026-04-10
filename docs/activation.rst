.. _activation-functions-label:

Overview of builtin activation functions
========================================

.. index:: ! activation function

Note that some of these :term:`functions <activation function>` are scaled differently from the canonical
versions you may be familiar with.  The intention of the scaling is to place
more of the functions' "interesting" behavior in the region :math:`\left[-1, 1\right] \times \left[-1, 1\right]`.

The implementation of these functions can be found in the :py:mod:`activations` module.

The following table summarizes the scaling, clamping, and non-canonical
behavior of the activation functions that differ from their textbook forms.
Input ``z`` is clamped to the given range before any output transform is
applied. Functions not listed below (``relu``, ``elu``, ``selu``, ``identity``,
``clamped``, ``abs``, ``hat``, ``square``, ``cube``) apply their canonical
transforms directly with no scaling or clamping.

+-------------+--------------------+----------------+------------------------------------------------+
| Function    | Input clamp        | Scaling        | Transform                                      |
+=============+====================+================+================================================+
| sigmoid     | ±60 after 5×z      | 5× input       | :math:`1 / (1 + e^{-5z})`                      |
+-------------+--------------------+----------------+------------------------------------------------+
| tanh        | ±60 after 2.5×z    | 2.5× input     | :math:`\tanh(2.5\,z)`                          |
+-------------+--------------------+----------------+------------------------------------------------+
| sin         | ±60 after 5×z      | 5× input       | :math:`\sin(5\,z)`                             |
+-------------+--------------------+----------------+------------------------------------------------+
| gauss       | ±3.4               | −5 in exponent | :math:`e^{-5 z^2}`                             |
+-------------+--------------------+----------------+------------------------------------------------+
| softplus    | ±60 after 5×z      | 5× in, 0.2× out| :math:`0.2 \log(1 + e^{5z})`                   |
+-------------+--------------------+----------------+------------------------------------------------+
| exp         | ±60                | none           | :math:`e^{z}`                                  |
+-------------+--------------------+----------------+------------------------------------------------+
| log         | floor at ``1e-7``  | none           | :math:`\log(\max(10^{-7}, z))` — non-positive  |
|             |                    |                | inputs yield :math:`\log(10^{-7}) \approx      |
|             |                    |                | -16.118` rather than ``ValueError``.           |
+-------------+--------------------+----------------+------------------------------------------------+
| inv         | none               | none           | :math:`1/z`, returning ``0.0`` on              |
|             |                    |                | ``ArithmeticError`` (e.g. division by zero     |
|             |                    |                | or overflow).                                  |
+-------------+--------------------+----------------+------------------------------------------------+
| lelu        | none               | none           | :math:`z` if :math:`z > 0`, otherwise          |
|             |                    |                | :math:`0.005\,z`. **Note: non-standard leak    |
|             |                    |                | coefficient** — the conventional leaky ReLU    |
|             |                    |                | uses ``0.01`` (e.g. PyTorch's                  |
|             |                    |                | ``nn.LeakyReLU`` default).                     |
+-------------+--------------------+----------------+------------------------------------------------+

abs
---

.. figure:: _static/activation-abs.png
   :scale: 50 %
   :alt: absolute value function


clamped
-------

.. figure:: _static/activation-clamped.png
   :scale: 50 %
   :alt: clamped linear function

cube
----

.. figure:: _static/activation-cube.png
   :scale: 50 %
   :alt: cubic function

exp
---

.. figure:: _static/activation-exp.png
   :scale: 50 %
   :alt: exponential function


gauss
-----

.. figure:: _static/activation-gauss.png
   :scale: 50 %
   :alt: gaussian function

hat
---

.. figure:: _static/activation-hat.png
   :scale: 50 %
   :alt: hat function

.. _identity-label:

identity
--------

.. figure:: _static/activation-identity.png
   :scale: 50 %
   :alt: identity function

inv
---

.. figure:: _static/activation-inv.png
   :scale: 50 %
   :alt: inverse function

log
---

.. figure:: _static/activation-log.png
   :scale: 50 %
   :alt: log function

relu
----

.. figure:: _static/activation-relu.png
   :scale: 50 %
   :alt: rectified linear function

elu
----

.. figure:: _static/activation-elu.png
   :scale: 50 %
   :alt: exponential rectified linear function

lelu
----

.. figure:: _static/activation-lelu.png
   :scale: 50 %
   :alt: leaky rectified linear function

selu
----

.. figure:: _static/activation-selu.png
   :scale: 50 %
   :alt: scaled exponential linear function

.. _sigmoid-label:

sigmoid
-------

.. figure:: _static/activation-sigmoid.png
   :scale: 50 %
   :alt: sigmoid function

sin
---

.. figure:: _static/activation-sin.png
   :scale: 50 %
   :alt: sine function

softplus
--------

.. figure:: _static/activation-softplus.png
   :scale: 50 %
   :alt: soft-plus function

square
------

.. figure:: _static/activation-square.png
   :scale: 50 %
   :alt: square function

tanh
----

.. figure:: _static/activation-tanh.png
   :scale: 50 %
   :alt: hyperbolic tangent function
