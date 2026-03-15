# Per-Node Time Constants in neat-python's CTRNN: What Changed and Why

## 1. Background

A Continuous-Time Recurrent Neural Network (CTRNN) models each node as a leaky integrator with state variable y_i and time constant tau_i. The underlying ODE is:

    tau_i * dy_i/dt = -y_i + f(bias_i + response_i * aggregation_j(w_ij * y_j))

where f is the activation function, w_ij are connection weights, and dt is the integration timestep. As of v2.0, this ODE is integrated using the exponential Euler (ETD1) method, which exactly integrates the linear decay term:

    decay_i = exp(-dt / tau_i)
    y_i(t + dt) = decay_i * y_i(t) + (1 - decay_i) * z_i

where z_i = f(bias_i + response_i * aggregation_j(w_ij * y_j)). This replaces the forward Euler method used prior to v2.0. The exponential Euler method is unconditionally stable for the linear decay regardless of the dt/tau_i ratio, which is critical when evolving per-node time constants (see Section 4).

The time constant tau_i controls how quickly node i responds to its inputs: small tau gives fast response (the node closely tracks its instantaneous input), while large tau gives slow response (the node integrates over a longer temporal window).

In the NEAT (NeuroEvolution of Augmenting Topologies) framework, the structure and parameters of the neural network are evolved simultaneously. For CTRNNs, the natural approach is to evolve tau_i as a per-node gene attribute alongside bias, response, activation function, and connection weights.

## 2. The v1.0 Limitation

In neat-python v1.0, the CTRNN implementation used a **single fixed time constant** shared by all nodes. The value was supplied as a parameter at network construction time:

```python
# v1.0 API
net = CTRNN.create(genome, config, time_constant)
```

Every node in the resulting network used the same tau, regardless of its role in the network. This was an implementation simplification inherited from the original neat-python codebase, not a deliberate modeling choice.

The consequence is that the network cannot represent dynamics that require multiple timescales. Consider a system where some state variables change rapidly (period ~0.1 seconds) while others modulate slowly (period ~5 seconds). A CTRNN with heterogeneous time constants can assign fast tau to nodes tracking the rapid variables and slow tau to nodes integrating the slow ones. A CTRNN with a single fixed tau is forced to use the same temporal resolution everywhere, which is necessarily a compromise: too fast and the slow dynamics are not captured, too slow and the fast dynamics are smoothed out.

### 2.1 Quantitative Impact

The limitation was characterized using a Lorenz attractor prediction task, comparing neat-python v1.0 (fixed tau = 1.0) against the Julia NeatEvolution library (per-node evolved tau in [0.01, 5.0]) on the same NEAT configuration and fitness function. The task requires predicting the next state of the Lorenz system given the current state, using a CTRNN evolved by NEAT.

On the most diagnostic condition (single-output z prediction with pre-computed product inputs), the results were:

| Implementation | Pearson correlation (z) | Mean squared error |
|---|---|---|
| Julia (per-node tau) | 0.94 | 0.023 |
| Python v1.0 (fixed tau = 1.0) | 0.51--0.61 | 0.15 |

The fixed time constant degraded correlation by approximately 40--45% in absolute terms and increased mean squared error by a factor of 6--7. This degradation was consistent across all experimental conditions tested: three input representations (base, pre-computed products, product aggregation) crossed with two output configurations (three-output and single-output), for a total of six conditions with six independent trials each. Full results are reported in `NEAT-PYTHON-LORENZ.md`.

The analysis concluded that the fixed time constant was the **dominant bottleneck** -- not a minor contributor, but the primary limiting factor -- and that the degradation was fundamental (not addressable by choosing a different fixed value of tau).

## 3. The v2.0 Change

### 3.1 Per-Node Evolvable Time Constant

In neat-python 2.0, `time_constant` is a per-node gene attribute of `DefaultNodeGene`, on the same footing as `bias`, `response`, `activation`, and `aggregation`. Each node in the genome carries its own tau_i, which is initialized from a configurable distribution and subject to mutation and crossover like any other numeric gene attribute.

The `CTRNN.create()` signature changed accordingly:

```python
# v2.0 API
net = CTRNN.create(genome, config)
```

The time constant is no longer a construction parameter. Instead, the factory method reads `node.time_constant` from each node gene in the genome and assigns it to the corresponding `CTRNNNodeEval`. The CTRNN integration loop (the `advance` method) is unchanged; it already used `ne.time_constant` per node internally. The change is in where that value comes from: a caller-supplied scalar in v1.0, versus the evolved genome in v2.0.

### 3.2 Configuration

The time constant is configured through the standard NEAT config file using the same parameter naming convention as other `FloatAttribute` genes (bias, response, weight). For a CTRNN application, the relevant section of the config file would include:

```ini
[DefaultGenome]
time_constant_init_mean      = 1.0
time_constant_init_stdev     = 0.5
time_constant_init_type      = gaussian
time_constant_max_value      = 5.0
time_constant_min_value      = 0.01
time_constant_mutate_power   = 0.1
time_constant_mutate_rate    = 0.5
time_constant_replace_rate   = 0.1
```

### 3.3 Backward Compatibility for Non-CTRNN Configurations

The `time_constant` attribute is defined with **inert defaults**: `init_stdev = 0.0`, `mutate_rate = 0.0`, `mutate_power = 0.0`, `replace_rate = 0.0`. When no `time_constant_*` parameters appear in the config file, every node is initialized to `time_constant = init_mean = 1.0` and this value never changes. Feedforward and discrete-time recurrent networks do not read the time constant, so the attribute has no effect on their behavior.

To support this, the config parser was relaxed so that parameters with non-`None` defaults are used silently when absent from the config file (rather than raising an error, as in v1.0). This affects only the new `time_constant` parameters; all pre-existing parameters either have `default=None` (requiring explicit specification, as before) or are already present in existing config files.

### 3.4 Genetic Distance

The `DefaultNodeGene.distance()` method was updated to include the time constant:

    d = |bias_1 - bias_2| + |response_1 - response_2| + |tau_1 - tau_2|

When time constants are not configured (all nodes have tau = 1.0 by default), this term contributes zero to the genetic distance, preserving the existing speciation behavior for non-CTRNN applications.

## 4. Numerical Stability Consideration

### 4.1 Exponential Euler (v2.0)

The exponential Euler method used in v2.0 is unconditionally stable for the linear decay term, regardless of the dt/tau_i ratio. For example, with dt = 0.1 and tau = 0.01, the decay factor is exp(-0.1/0.01) = exp(-10) ≈ 0.000045, meaning the node responds almost instantaneously to its input — the state effectively jumps to z_i each step. This is physically correct behavior (a very fast node tracks its input closely), not a numerical instability.

The stability guarantee applies only to the linear decay. The nonlinear forcing term z_i = f(bias + response * aggregation(w_ij * y_j)) can still produce large values if weights are large, but this is bounded by the activation function (e.g., tanh saturates at ±1, sigmoid at [0,1]). Unbounded activation functions (identity, relu) with large weights can produce growing trajectories, but this is a modeling issue rather than a numerical stability issue.

As a result, the `time_constant_min_value` constraint is relaxed compared to the forward Euler era. Users can safely set `time_constant_min_value` well below the integration timestep without risking numerical blowup.

### 4.2 Forward Euler (prior to v2.0)

Prior to v2.0, the forward Euler update

    y_i(t + dt) = y_i(t) + (dt / tau_i) * (-y_i(t) + z_i)

was stable only when dt / tau_i was of order 1 or smaller. When tau_i was much smaller than dt, the factor dt / tau_i became large and the integration oscillated with exponentially growing amplitude. For example, with dt = 0.1 and tau = 0.01, the factor dt / tau = 10. The state at each step was multiplied by approximately (1 - dt/tau) = -9, producing rapid divergence.

The recommended workaround was to guard against this in the fitness function:

```python
if any(math.isnan(v) or math.isinf(v) or abs(v) > 1e10 for v in output):
    return PENALTY_FITNESS
```

This workaround is no longer necessary with the exponential Euler method, though it remains harmless if present in existing code.

## 5. Quantitative Improvement

The Lorenz attractor prediction task was repeated under the same conditions as the v1.0 baseline, with only the time constant treatment changed. Three independent trials per condition were run.

### 5.1 Single-Output Z Prediction

| Input mode | v1.0 z corr | v2.0 z corr | v1.0 MSE | v2.0 MSE |
|---|---|---|---|---|
| Base (x, y, z) | 0.32--0.44 | 0.64--0.66 | 0.16--0.17 | 0.107--0.109 |
| Products (x, y, z, xy, xz, yz) | 0.51--0.61 | 0.78--0.84 | 0.15 | 0.056--0.073 |
| Product aggregation | 0.29--0.40 | 0.57--0.65 | 0.16--0.17 | 0.108--0.127 |

Correlation improves by 50--80% (absolute) and mean squared error decreases by a factor of 1.3--2.7 across conditions.

### 5.2 Three-Output Prediction (x, y, z Simultaneously)

| Input mode | v1.0 best single-var corr | v2.0 best single-var corr | v1.0 MSE | v2.0 MSE |
|---|---|---|---|---|
| Base | x: 0.49--0.71 | x: 0.81--0.85 | 0.15--0.16 | 0.13--0.14 |
| Products | x: 0.00--0.72 | x/z: 0.76--0.92 | 0.14--0.16 | 0.12--0.14 |
| Product aggregation | x: 0.54--0.71 | x: 0.80--0.85 | 0.15--0.16 | 0.13--0.14 |

### 5.3 Comparison with Julia Reference Implementation

| Condition | Julia (per-node tau) | Python v2.0 (per-node tau) | Python v1.0 (fixed tau) |
|---|---|---|---|
| z-only base (correlation) | 0.83--0.86 | 0.64--0.66 | 0.32--0.44 |
| z-only products (correlation) | 0.94 | 0.78--0.84 | 0.51--0.61 |
| z-only products (MSE) | 0.023 | 0.056--0.073 | 0.15 |
| 3-output x base (correlation) | 0.84--0.96 | 0.81--0.85 | 0.49--0.71 |

Version 2.0 closes approximately half the gap between v1.0 and the Julia reference in z-only mode, and nearly matches Julia for single-variable prediction in three-output mode. The remaining difference is attributable to other implementation differences between the two NEAT libraries (speciation algorithm details, crossover mechanics) rather than the time constant treatment.

Full trial-level results are reported in `NEAT-PYTHON2-LORENZ.md`.

## 6. Migration Guide

### 6.1 Code Changes Required

Any code that calls `CTRNN.create()` with a time constant argument must be updated:

```python
# v1.0 (no longer valid)
net = CTRNN.create(genome, config, time_constant=0.01)

# v2.0
net = CTRNN.create(genome, config)
```

The old call will produce: `TypeError: create() takes 2 positional arguments but 3 were given`.

### 6.2 Config File Changes Required

CTRNN config files should add `time_constant_*` parameters to the `[DefaultGenome]` section. To replicate the v1.0 behavior of a fixed time constant (for example, tau = 0.01 for all nodes), use:

```ini
time_constant_init_mean      = 0.01
time_constant_init_stdev     = 0.0
time_constant_mutate_rate    = 0.0
time_constant_replace_rate   = 0.0
```

To enable evolution of per-node time constants, provide nonzero `init_stdev`, `mutate_rate`, and `mutate_power`, and set `min_value` and `max_value` to appropriate bounds for the problem's timescales.

### 6.3 Config Files That Do Not Use CTRNNs

No changes required. Feedforward and discrete-time recurrent network configurations work without modification. The time constant attribute uses inert defaults when not explicitly configured.

## 7. References

- Beer, R. D. (1995). On the dynamics of small continuous-time recurrent neural networks. *Adaptive Behavior*, 3(4), 469--509.
- Stanley, K. O. & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. *Evolutionary Computation*, 10(2), 99--127.
- `NEAT-PYTHON-LORENZ.md` -- v1.0 experimental results with fixed time constant.
- `NEAT-PYTHON2-LORENZ.md` -- v2.0 experimental results with per-node time constants.
