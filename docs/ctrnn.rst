Continuous-time recurrent neural network implementation
=======================================================

.. index:: ! continuous-time

.. index:: recurrent

.. index:: ! ctrnn

The default :term:`continuous-time` :term:`recurrent` neural network (CTRNN) :py:mod:`implementation <ctrnn>` in neat-python
is modeled as a system of ordinary differential equations, with neuron potentials as the dependent variables.

:math:`\tau_i \frac{d y_i}{dt} = -y_i + f_i\left(\beta_i + \sum\limits_{j \in A_i} w_{ij} y_j\right)`


Where:

* :math:`\tau_i` is the time constant of neuron :math:`i`.
* :math:`y_i` is the potential of neuron :math:`i`.
* :math:`f_i` is the :term:`activation function` of neuron :math:`i`.
* :math:`\beta_i` is the :term:`bias` of neuron :math:`i`.
* :math:`A_i` is the set of indices of neurons that provide input to neuron :math:`i`.
* :math:`w_{ij}` is the :term:`weight` of the :term:`connection` from neuron :math:`j` to neuron :math:`i`.

The time evolution of the network is computed using the exponential Euler (ETD1) method, which
integrates the linear decay term exactly:

:math:`y_i(t+\Delta t) = e^{-\Delta t / \tau_i} \cdot y_i(t) + \left(1 - e^{-\Delta t / \tau_i}\right) \cdot z_i`

where :math:`z_i = f_i\left(\beta_i + \rho_i \sum\limits_{j \in A_i} w_{ij} y_j\right)` is the
activated output and :math:`\rho_i` is the response multiplier of neuron :math:`i`.

This method is **unconditionally stable** for the linear decay part regardless of the ratio
:math:`\Delta t / \tau_i`. Forward Euler (used in versions prior to 2.0) required
:math:`\Delta t < 2 \tau_i` for stability, which was problematic when evolving per-node time
constants — nodes with small :math:`\tau_i` relative to the integration timestep would produce
divergent trajectories.

.. note::
   The exponential Euler method holds the nonlinear term :math:`z_i` constant over each timestep
   (the same assumption as forward Euler). The accuracy advantage comes from exactly integrating
   the linear decay :math:`-y_i / \tau_i`, which is the dominant term for stiff systems where
   :math:`\tau_i \ll \Delta t`.

GPU-Accelerated Evaluation
--------------------------

For large populations, CTRNN evaluation can be accelerated on GPU using the optional ``neat.gpu``
module. This requires CuPy (install via ``pip install 'neat-python[gpu]'``).

The GPU evaluator uses the same exponential Euler integration method as the CPU implementation.
Variable-topology genomes are packed into fixed-size padded tensors and evaluated in a single
batched operation across the entire population.

.. code-block:: python

   from neat.gpu.evaluator import GPUCTRNNEvaluator

   def input_fn(t, dt):
       """Return input values at time t. Shape: [num_inputs]."""
       return [math.sin(2 * math.pi * t), math.cos(2 * math.pi * t)]

   def fitness_fn(output_trajectory):
       """Compute fitness from output trajectory. Shape: [num_steps, num_outputs]."""
       return float(output_trajectory[-1, 0])

   evaluator = GPUCTRNNEvaluator(dt=0.01, t_max=1.0,
                                  input_fn=input_fn, fitness_fn=fitness_fn)
   winner = population.run(evaluator.evaluate, n=300)

**Constraints:**

* Only ``sum`` aggregation is supported (required for batched matrix-vector multiply).
* Supported activation functions: sigmoid, tanh, relu, identity, clamped, elu, softplus, sin,
  gauss, abs, square.
* Genomes using unsupported aggregation or activation functions will raise ``ValueError`` at
  evaluation time.
