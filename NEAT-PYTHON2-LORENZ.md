# Evolving CTRNNs for Lorenz System Prediction Using neat-python 2.0

## 1. Overview

This document reports the results of repeating the Lorenz CTRNN experiment with neat-python 2.0, which introduces **per-node evolvable time constants**. The v1.0 experiment (`NEAT-PYTHON-LORENZ.md`) identified the fixed time constant as the dominant bottleneck, degrading correlation by ~2x and MSE by 2--3x compared to the Julia NeatEvolution implementation. Version 2.0 removes this limitation.

The same 6-condition protocol is used: base/products/product-agg x 3-output/z-only, with 3 independent trials per condition. All other NEAT parameters are unchanged from v1.0.

## 2. What Changed in v2.0

### 2.1 Per-Node Time Constants

`time_constant` is now a per-node evolvable attribute of `DefaultNodeGene`. Each node has its own tau, initialized from a Gaussian distribution and subject to mutation. The `CTRNN.create()` method no longer accepts a fixed time constant parameter; it reads tau from each node's genome.

For this experiment, the time constant configuration matches the Julia NeatEvolution settings:

| Parameter | Value |
|-----------|-------|
| `time_constant_init_mean` | 1.0 |
| `time_constant_init_stdev` | 0.5 |
| `time_constant_min_value` | 0.01 |
| `time_constant_max_value` | 5.0 |
| `time_constant_mutate_rate` | 0.5 |
| `time_constant_mutate_power` | 0.1 |
| `time_constant_replace_rate` | 0.1 |

### 2.2 Numerical Stability Guard

With per-node time constants, some nodes can have small tau values (near 0.01). When the data timestep dt = 0.1 and tau = 0.01, the Euler update factor dt/tau = 10, which drives the explicit Euler integration into instability: state values oscillate with exponentially growing amplitude, producing large finite values that overflow float64 when squared in the fitness function.

The v1.0 NaN/Inf guard was extended to also catch large finite values:

```python
if any(math.isnan(v) or math.isinf(v) or abs(v) > 1e10 for v in output):
    return PENALTY_FITNESS
```

Genomes that produce numerically unstable outputs are assigned the penalty fitness (-10.0), allowing evolution to select against unstable time constant configurations. This is the correct evolutionary approach: rather than clamping or modifying the integration, let selection pressure eliminate configurations where dt/tau is too large.

### 2.3 All Other Parameters Unchanged

Population size (150), generations (300), activation functions (tanh, sigmoid, relu), weight/bias ranges ([-5, 5]), speciation parameters, and all other NEAT settings are identical to v1.0.

## 3. Results

### 3.1 Three-Output Mode (predicting x, y, z simultaneously)

Individual trial results:

| Mode | Trial | x corr | y corr | z corr | Test MSE | Nodes | Time |
|------|-------|--------|--------|--------|----------|-------|------|
| base | 1 | 0.8181 | 0.0000 | 0.0000 | 0.135 | 5 | 25.8s |
| base | 2 | 0.8469 | 0.0000 | 0.0000 | 0.132 | 9 | 36.0s |
| base | 3 | 0.8102 | 0.0000 | 0.0000 | 0.136 | 6 | 32.9s |
| products | 1 | 0.0000 | 0.0000 | 0.7597 | 0.142 | 8 | 35.6s |
| products | 2 | 0.0000 | 0.0000 | 0.7833 | 0.139 | 9 | 37.2s |
| products | 3 | 0.9158 | 0.0000 | 0.0000 | 0.124 | 6 | 33.6s |
| product-agg | 1 | 0.7952 | 0.0000 | 0.0000 | 0.137 | 3 | 26.3s |
| product-agg | 2 | 0.8488 | 0.0000 | 0.0000 | 0.132 | 11 | 33.5s |
| product-agg | 3 | 0.8309 | 0.0000 | 0.0000 | 0.136 | 6 | 31.0s |

Summary ranges:

| Mode | x corr | y corr | z corr | Test MSE | Nodes | Time |
|------|--------|--------|--------|----------|-------|------|
| base | 0.81--0.85 | 0.00 | 0.00 | 0.13--0.14 | 5--9 | 26--36s |
| products | 0.00--0.92 | 0.00 | 0.00--0.78 | 0.12--0.14 | 6--9 | 34--37s |
| product-agg | 0.80--0.85 | 0.00 | 0.00 | 0.13--0.14 | 3--11 | 26--34s |

### 3.2 Z-Only Mode (predicting z alone)

Individual trial results:

| Mode | Trial | z corr | Test MSE | Nodes | Time |
|------|-------|--------|----------|-------|------|
| base | 1 | 0.6557 | 0.107 | 9 | 31.6s |
| base | 2 | 0.6495 | 0.108 | 6 | 32.4s |
| base | 3 | 0.6404 | 0.109 | 10 | 29.0s |
| products | 1 | 0.7790 | 0.073 | 1 | 25.5s |
| products | 2 | 0.7846 | 0.071 | 1 | 29.5s |
| products | 3 | 0.8359 | 0.056 | 2 | 29.0s |
| product-agg | 1 | 0.6536 | 0.111 | 4 | 28.5s |
| product-agg | 2 | 0.6515 | 0.108 | 4 | 30.7s |
| product-agg | 3 | 0.5696 | 0.127 | 6 | 31.8s |

Summary ranges:

| Mode | z corr | Test MSE | Nodes | Time |
|------|--------|----------|-------|------|
| base | 0.64--0.66 | 0.107--0.109 | 6--10 | 29--32s |
| products | 0.78--0.84 | 0.056--0.073 | 1--2 | 26--30s |
| product-agg | 0.57--0.65 | 0.108--0.127 | 4--6 | 29--32s |

## 4. Comparison with v1.0 (Fixed Time Constant)

### 4.1 Z-Only Mode

| Mode | v1.0 z corr | v2.0 z corr | v1.0 MSE | v2.0 MSE |
|------|-------------|-------------|----------|----------|
| base | 0.32--0.44 | 0.64--0.66 | 0.16--0.17 | 0.107--0.109 |
| products | 0.51--0.61 | 0.78--0.84 | 0.15 | 0.056--0.073 |
| product-agg | 0.29--0.40 | 0.57--0.65 | 0.16--0.17 | 0.108--0.127 |

Per-node time constants improve z-only correlation by 50--80% across all conditions, and reduce MSE by 30--60%.

### 4.2 Three-Output Mode

| Mode | v1.0 best single corr | v2.0 best single corr | v1.0 MSE | v2.0 MSE |
|------|----------------------|----------------------|----------|----------|
| base | x: 0.49--0.71 | x: 0.81--0.85 | 0.15--0.16 | 0.13--0.14 |
| products | x: 0.00--0.72 | x/z: 0.76--0.92 | 0.14--0.16 | 0.12--0.14 |
| product-agg | x: 0.54--0.71 | x: 0.80--0.85 | 0.15--0.16 | 0.13--0.14 |

The best single-variable correlation in 3-output mode improves substantially. In v1.0, evolution typically learned a weak approximation of x (0.49--0.72); in v2.0, the learned variable reaches 0.80--0.92 correlation.

### 4.3 Comparison with Julia NeatEvolution

| Condition | Julia | Python v1.0 | Python v2.0 |
|-----------|-------|-------------|-------------|
| z-only base | 0.83--0.86 | 0.32--0.44 | 0.64--0.66 |
| z-only products | 0.94 | 0.51--0.61 | 0.78--0.84 |
| z-only product-agg | 0.94 | 0.29--0.40 | 0.57--0.65 |
| 3-out x (base) | 0.84--0.96 | 0.49--0.71 | 0.81--0.85 |
| 3-out x (products) | 0.84--0.96 | 0.00--0.72 | 0.00--0.92 |

Python v2.0 closes roughly half the gap to Julia in z-only mode, and nearly matches Julia for x prediction in 3-output mode. The remaining gap is discussed in Section 6.

## 5. Analysis

### 5.1 Per-Node Time Constants Provide Major Improvement

The hypothesis from the v1.0 writeup -- that the fixed time constant was the dominant bottleneck -- is confirmed. Every condition improves substantially:

- **z-only base**: correlation 0.32--0.44 -> 0.64--0.66 (approximately +70%)
- **z-only products**: correlation 0.51--0.61 -> 0.78--0.84 (approximately +50%), MSE 0.15 -> 0.056--0.073 (approximately 2--3x improvement)
- **z-only product-agg**: correlation 0.29--0.40 -> 0.57--0.65 (approximately +65%)

The z-only products condition shows the most dramatic MSE improvement: the best v2.0 trial achieves MSE 0.056, which is within the range of Julia's base condition (0.055--0.069), despite Julia's products condition achieving 0.023.

### 5.2 Fitness Dilution Remains the Dominant Effect in 3-Output Mode

In 3-output mode, every v2.0 trial learns exactly one variable well (correlation 0.76--0.92) while the other two remain at zero. This is the same fitness dilution pattern seen in v1.0 and Julia: averaging MSE across three outputs means that evolution can reduce total error by improving one output to track x's oscillations (or occasionally z's), while the other two outputs learn the unconditional mean.

The pattern of which variable gets learned varies:
- **base and product-agg**: consistently learn x (all 6 trials)
- **products**: learns z in 2/3 trials, x in 1/3 trial

This makes sense. In base mode, x is the easiest variable (largest relative variance in the normalized data, most autocorrelated). Products mode provides xy and xz as additional inputs, which are directly informative about z (since dz/dt = xy - bz), making z suddenly accessible.

### 5.3 Product Inputs Help, Product Aggregation Partially Recovers

In z-only mode, the ranking is: products > base approximately product-agg, the same qualitative ordering as v1.0. However, the gap between product-agg and base has narrowed:

| v1.0 | v2.0 |
|------|------|
| products: 0.51--0.61 | products: 0.78--0.84 |
| base: 0.32--0.44 | base: 0.64--0.66 |
| product-agg: 0.29--0.40 | product-agg: 0.57--0.65 |

In v1.0, product-agg performed *worse* than base (0.29--0.40 vs 0.32--0.44). In v2.0, product-agg performs comparably to base (0.57--0.65 vs 0.64--0.66), though still worse. Per-node time constants partially rehabilitate product aggregation by providing the compensating temporal expressiveness that multiplicative nodes need (as hypothesized in the v1.0 analysis), but the benefit is modest.

In Julia, product-agg matches products exactly (both 0.94). The fact that Python v2.0 product-agg (0.57--0.65) still substantially underperforms products (0.78--0.84) indicates that other differences between the Python and Julia implementations affect product-agg more than they affect the other modes.

### 5.4 Network Complexity

Evolved network sizes are comparable to v1.0:

| Condition | v1.0 nodes | v2.0 nodes |
|-----------|-----------|-----------|
| z-only base | 2--10 | 6--10 |
| z-only products | 1--5 | 1--2 |
| z-only product-agg | 4--8 | 4--6 |
| 3-out base | 3--6 | 5--9 |

The z-only products networks are notably minimal: 1--2 nodes in v2.0 (matching Julia's 1--3 nodes). With the bilinear input xy provided directly, the z prediction problem is essentially linear, and evolution discovers this.

### 5.5 Numerical Stability and the dt/tau Constraint

The overflow guard (Section 2.2) was essential. Without it, all 6 initial runs crashed. The root cause is fundamental: explicit Euler integration of the CTRNN state update requires dt/tau < ~1 for stability, but with tau_min = 0.01 and dt = 0.1, dt/tau can reach 10.

The evolutionary approach (penalize unstable genomes) is preferable to engineering fixes (clamping tau or using implicit integration) because:
1. It lets evolution discover the useful range of time constants naturally.
2. It avoids introducing bias into the dynamics.
3. It adds no computational overhead to stable genomes.

In practice, the penalty is applied rarely after the first few generations, as selection pressure quickly eliminates genomes with very small time constants relative to the data timestep.

## 6. Remaining Gap to Julia

Python v2.0 closes roughly half the gap to Julia. The remaining differences likely stem from:

1. **Speciation and reproduction details**: Julia NeatEvolution and neat-python implement NEAT's speciation algorithm slightly differently (separate disjoint/excess coefficients vs combined, different stagnation handling). These affect the evolutionary dynamics and population diversity.

2. **300 generations may be insufficient**: The Julia results were also at 300 generations, but Julia's faster per-evaluation throughput means the evolutionary search may traverse more of the fitness landscape per generation (larger effective population due to less stagnation).

3. **Product-agg gap**: The largest remaining gap is in product-agg (v2.0: 0.57--0.65 vs Julia: 0.94). This suggests that neat-python's evolutionary operators have difficulty optimizing multiplicative nodes even with per-node time constants. The crossover and mutation dynamics for product-aggregation nodes may interact unfavorably with Python's speciation scheme.

4. **Integration accuracy**: Both implementations use explicit Euler, but the penalty guard in the Python version may eliminate some genome configurations that the Julia version can evaluate (if Julia handles the numerical instability differently).

## 7. Summary

### 7.1 Key Findings

1. **Per-node time constants improve all conditions substantially**, confirming the v1.0 hypothesis. Z-only correlation improves 50--80%, MSE improves 30--60%.

2. **The improvement is largest for products z-only** (MSE: 0.15 -> 0.056--0.073), which approaches Julia's base-condition performance.

3. **Fitness dilution remains the primary obstacle** in 3-output mode. Per-node time constants do not help here because the problem is in the fitness signal, not the network's expressiveness.

4. **Product aggregation is partially rehabilitated** by per-node time constants (no longer worse than base, as in v1.0) but still substantially underperforms pre-computed product inputs.

5. **3-output x correlation now matches Julia** (v2.0: 0.81--0.92 vs Julia: 0.84--0.96), demonstrating that per-node time constants were the primary bottleneck for single-variable prediction accuracy.

### 7.2 Cross-Version Summary Table

**Z-only mode (best condition: products):**

| Implementation | z correlation | z MSE |
|---------------|--------------|-------|
| Julia NeatEvolution | 0.94 | 0.023 |
| Python v2.0 (per-node tau) | 0.78--0.84 | 0.056--0.073 |
| Python v1.0 (fixed tau=1.0) | 0.51--0.61 | 0.15 |

**3-output mode (best single-variable correlation):**

| Implementation | Best corr | MSE |
|---------------|-----------|-----|
| Julia NeatEvolution | x: 0.84--0.96 | 0.10--0.12 |
| Python v2.0 (per-node tau) | x: 0.81--0.92 | 0.12--0.14 |
| Python v1.0 (fixed tau=1.0) | x: 0.49--0.72 | 0.14--0.16 |

### 7.3 Recommendation Update

The v1.0 recommendation to "consider using RecurrentNetwork instead of CTRNN" is no longer necessary. With per-node time constants, CTRNN is the appropriate choice for dynamical systems prediction. Users should configure the `time_constant_*` parameters in their config file to match the timescales of their problem, and ensure `time_constant_min_value` is not too small relative to the data timestep to avoid numerical instability.
