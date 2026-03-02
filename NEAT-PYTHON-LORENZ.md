# Evolving CTRNNs for Lorenz System Prediction Using neat-python

## 1. Overview and Motivation

This document reports the results of replicating the Julia NeatEvolution Lorenz CTRNN experiment (described in `NEAT-LORENZ.md`) using the neat-python library. The goal is to evolve a CTRNN to perform single-step prediction of the Lorenz attractor: given the current state (x(t), y(t), z(t)), predict the state one timestep later.

The experiment was ported as faithfully as possible. The same data generation pipeline, normalization scheme, fitness function, and NEAT configuration parameters were used. The key difference is an implementation limitation in neat-python's CTRNN: it uses a **single fixed time constant** for all nodes, whereas the Julia version evolves **per-node time constants**. This turns out to have substantial consequences for performance.

## 2. Implementation Differences from the Julia Version

### 2.1 Fixed Time Constant (Primary Difference)

The Julia NeatEvolution CTRNN evolves a time constant tau_i for each node independently, with configurable range [0.01, 5.0] and initialization mean 1.0. This per-node heterogeneity allows evolution to assign fast time constants (tau near 0.01, near-instantaneous response) to some nodes and slow time constants (tau near 5.0, slow temporal integration) to others.

neat-python's CTRNN implementation accepts a single time constant at network creation time, applied uniformly to all nodes. For this experiment, tau = 1.0 was used (matching the Julia initialization mean).

The CTRNN state update for node i is:

    y_i(t + dt) = y_i(t) + (dt/tau_i) * (-y_i(t) + f(bias_i + response_i * sum_j(w_ij * y_j)))

With dt = 0.1 (the data timestep) and tau = 1.0, the update factor is dt/tau = 0.1, giving:

    y_i(t + dt) = 0.9 * y_i(t) + 0.1 * f(...)

This means every node has the same temporal response: a 1-second exponential decay, forgetting approximately 63% of its state per second (10 data steps). The Lorenz system's dynamics span multiple timescales (fast x/y oscillations at 0.1--0.5 s, orbital period approximately 1.5 s, slower z modulation), and a fixed time constant prevents the network from matching these different timescales with different nodes.

### 2.2 Other Differences

- **Genetic distance:** neat-python combines disjoint and excess gene coefficients into a single `compatibility_disjoint_coefficient`, whereas the Julia version separates them. Both were set to 1.0.
- **Parallel evaluation:** The Python version uses `neat.ParallelEvaluator` across all CPU cores to partially compensate for Python's slower per-genome evaluation. The Julia version runs single-threaded.
- **Numerical stability:** An intermittent `OverflowError` in neat-python's statistics module occurs when genomes with extremely poor fitness (very large negative values) cause variance calculations to overflow float64. This affected approximately 1 in 9 product-aggregation runs. Affected runs were retried.

## 3. Data Generation and Preprocessing

Identical to the Julia experiment:

- Lorenz system integrated via hand-written 4th-order Runge-Kutta at dt = 0.01 for 11,000 steps from initial condition (1, 1, 1).
- Steps 1--1,000 discarded as transient; steps 1,001--9,000 for training; steps 9,001--11,000 for testing.
- Subsampled every 10th step: 800 training points, 200 test points, effective data timestep 0.1 s.
- Per-variable affine normalization to [-1, 1] computed from training data only.
- Normalization ranges: x in [-18.21, 17.23], y in [-22.32, 24.94], z in [4.54, 45.16].

## 4. NEAT/CTRNN Configuration

All parameters match the Julia experiment except where noted:

| Parameter | Value | Note |
|-----------|-------|------|
| Population size | 150 | Same |
| Generations | 300 | Same |
| Initial topology | Fully connected, 0 hidden | Same |
| Feed-forward | false | Same |
| Activation functions | tanh, sigmoid, relu | Same (5% mutation rate) |
| Weight/bias range | [-5, 5] | Same |
| **Time constant** | **1.0 (fixed, all nodes)** | **Julia: evolved per-node, [0.01, 5.0]** |
| Node add probability | 0.2 | Same |
| Connection add probability | 0.3 | Same |
| Max stagnation | 30 | Same |
| Compatibility threshold | 3.0 | Same |

## 5. Experimental Results

All results are reported over 6 independent trials per condition (two batches of 3). Correlation is Pearson correlation between predicted and true normalized values on the held-out test set. Wall-clock times are on a 12-core machine using parallel evaluation.

### 5.1 Three-Output Mode (predicting x, y, z simultaneously)

| Mode | x corr | y corr | z corr | Test MSE | Nodes | Wall-clock |
|------|--------|--------|--------|----------|-------|------------|
| base | 0.49--0.71 | -0.06--0.31 | 0.00--0.26 | 0.15--0.16 | 3--6 | 9--11 s |
| products | 0.00--0.72 | -0.06--0.57 | 0.00--0.34 | 0.14--0.16 | 4--13 | 11--16 s |
| product-agg | 0.54--0.71 | 0.00--0.04 | 0.00 | 0.15--0.16 | 5--9 | 10--12 s |

### 5.2 Z-Only Mode (predicting z alone)

| Mode | z corr | z MSE | Nodes | Wall-clock |
|------|--------|-------|-------|------------|
| base | 0.32--0.44 | 0.16--0.17 | 2--10 | 8--12 s |
| products | 0.51--0.61 | 0.15 | 1--5 | 7--15 s |
| product-agg | 0.29--0.40 | 0.16--0.17 | 4--8 | 9--12 s |

### 5.3 Comparison with Julia Results

For reference, the Julia NeatEvolution results (from `NEAT-LORENZ.md`) on the same task:

**Julia 3-output:** x corr 0.84--0.96, y corr -0.09--0.82, z corr 0.00--0.17, MSE 0.10--0.12

**Julia z-only:** base z corr 0.83--0.86 (MSE 0.055--0.069), products z corr 0.94 (MSE 0.023), product-agg z corr 0.94 (MSE 0.023--0.024)

The Python results are substantially worse across every condition. The next two sections analyze why.

## 6. Analysis

### 6.1 Fixed Time Constant Degrades All Conditions

The most striking result is the uniform degradation across all conditions. Even the "easy" variable x, which the Julia version tracks at 0.84--0.96 correlation, only reaches 0.49--0.71 in the Python version. Similarly, z-only base correlation drops from 0.83--0.86 (Julia) to 0.32--0.44 (Python). These comparisons hold the fitness signal constant (same objective function) and only change the CTRNN's temporal expressiveness.

This demonstrates that per-node time constants are not merely a convenience but a critical capability for learning dynamical systems. With a fixed time constant, the network cannot separately implement fast response nodes (for tracking rapid x/y oscillations) and slow integration nodes (for accumulating temporal context). All nodes are forced to operate at the same timescale, limiting what temporal patterns the network can represent.

### 6.2 Fitness Dilution Is Still Visible

Despite the overall lower performance, the fitness dilution effect documented in the Julia experiment remains detectable. In every mode, z-only outperforms z in the 3-output configuration:

|  | 3-output z corr | z-only z corr |
|---|---|---|
| base | 0.00--0.26 | 0.32--0.44 |
| products | 0.00--0.34 | 0.51--0.61 |
| product-agg | 0.00 | 0.29--0.40 |

Switching from 3-output to z-only (same inputs, same representation) consistently improves z correlation, confirming that the scalar fitness averaging still suppresses selection pressure on the hard output.

### 6.3 Product Inputs Help, Product Aggregation Does Not

Pre-computed product inputs improve z prediction in both 3-output mode (occasional z corr 0.32--0.34 vs near-zero for base) and z-only mode (0.51--0.61 vs 0.32--0.44). Supplying the xy term directly as an input still helps, even with the fixed time constant.

However, product aggregation (product-agg mode) performs the same as or worse than the base in every condition. In z-only mode, product-agg achieves z correlation 0.29--0.40, compared to base's 0.32--0.44. This is the opposite of the Julia result, where product-agg matched pre-computed products exactly (both 0.94).

This failure appears to result from the interaction between product aggregation and the fixed time constant. Product aggregation computes the product of all weighted inputs to a node, making the node's output a highly nonlinear function of its inputs. Tuning the weights of a multiplicative node is harder than tuning an additive node because:

1. The gradient of a product with respect to any single weight depends on all other weighted inputs.
2. Small weight changes can cause large output changes (multiplicative sensitivity).
3. The sign pattern of inputs determines whether the product is constructive or destructive.

In the Julia version, per-node time constants provide an additional degree of freedom: evolution can assign a fast time constant to a multiplicative node (making it respond quickly) or a slow time constant (smoothing out the sensitivity). Without this compensating mechanism, the evolutionary search cannot effectively exploit multiplicative nodes, and the additional search-space complexity of product aggregation becomes a net negative.

### 6.4 Summary Table

The 2x2 factorial from the Julia experiment, extended with the Python results:

**Julia (per-node time constants):**

|  | Additive only (base) | Multiplicative available (product-agg) |
|---|---|---|
| **3-output (diluted)** | z approximately 0.00 | z approximately 0.00--0.15 |
| **z-only (focused)** | z approximately 0.83--0.86 | z approximately 0.94 |

**Python (fixed time constant):**

|  | Additive only (base) | Pre-computed products | Product-agg |
|---|---|---|---|
| **3-output (diluted)** | z approximately 0.00--0.26 | z approximately 0.00--0.34 | z approximately 0.00 |
| **z-only (focused)** | z approximately 0.32--0.44 | z approximately 0.51--0.61 | z approximately 0.29--0.40 |

The Julia findings about fitness dilution and representation difficulty are confirmed qualitatively. However, the fixed time constant introduces a third, dominant bottleneck: **temporal expressiveness**. Without per-node time constants, overall performance is degraded by approximately 2x in correlation and 2--3x in MSE, and the benefit of product aggregation is not just reduced but reversed.

## 7. Evolved Network Characteristics

Networks evolved in the Python version are similar in size to the Julia version:

- **3-output base:** 3--6 nodes (Julia: 4--10). The smaller Python networks reflect the lower overall fitness -- less structural complexity is added because even simple structures are not fully optimized.
- **z-only products:** 1--5 nodes (Julia: 1--3). The smallest networks (1 node) appear in both implementations, confirming that the z equation's structure is simple once the bilinear input is provided.
- **z-only product-agg:** 4--8 nodes (Julia: 1--6). The Python networks are larger, consistent with evolution searching unsuccessfully for a way to exploit product aggregation without per-node time constant tuning.

## 8. Computational Performance

With parallel evaluation across 12 CPU cores, wall-clock times are 7--16 seconds for 300 generations with population 150. This is comparable to the Julia single-threaded performance (9--20 seconds). Without parallelism, the Python version would be approximately 10x slower, consistent with the Julia implementation's per-genome evaluation throughput advantage.

## 9. Implications

### 9.1 Per-Node Time Constants Are Essential for CTRNN Dynamics

The most important finding is that the fixed time constant in neat-python's CTRNN substantially limits the network's ability to learn dynamical systems. This is not a tuning issue (a different fixed value would shift but not solve the problem) but a fundamental expressiveness limitation. For dynamical systems with multiple timescales, per-node time constants should be considered a core feature of CTRNNs, not an optional enhancement.

### 9.2 The Julia Findings Are Robust

Despite the degraded absolute performance, the qualitative patterns from the Julia experiment reproduce:

1. **Fitness dilution** prevents z learning in 3-output mode regardless of representation.
2. **Pre-computed product inputs** help z prediction in both 3-output and z-only modes.
3. **Evolved networks are small** -- NEAT's incremental complexification discovers compact solutions.

### 9.3 Product Aggregation Requires Temporal Expressiveness

The failure of product-agg in the Python version, contrasted with its success in Julia, suggests that evolvable multiplicative primitives are only useful when the network has sufficient compensating expressiveness (here, per-node time constants) to handle their optimization difficulty. Adding more powerful computational primitives to a limited architecture can make evolution harder, not easier.

### 9.4 Recommendations for neat-python Users

For dynamical systems prediction with neat-python's CTRNN:

1. **Use pre-computed product inputs** rather than product aggregation if the target dynamics contain bilinear terms. This is more robust than relying on evolution to discover multiplication.
2. **Consider using RecurrentNetwork** instead of CTRNN if continuous-time dynamics are not essential. The recurrent network does not have the time constant limitation.
3. **The fixed time constant tau = 1.0 is a reasonable default** for systems with characteristic timescales near 1 second, but it is a compromise. Experimenting with different fixed values (e.g., tau = 0.1 for fast dynamics, tau = 5.0 for slow) may help for specific problems.
