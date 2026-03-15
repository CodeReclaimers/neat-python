# GPU Evaluation Design Notes

## 1. Extracting Parameters from neat-python Genomes

### CTRNN Genomes

Each CTRNN genome uses `DefaultGenome` with `DefaultNodeGene` nodes and
`DefaultConnectionGene` connections. The relevant per-node attributes are:

| Attribute       | Source                    | Notes |
|-----------------|---------------------------|-------|
| `bias`          | `node.bias`               | Additive bias term |
| `response`      | `node.response`           | Multiplicative gain on aggregated input |
| `time_constant` | `node.time_constant`      | τ_i in the CTRNN ODE |
| `activation`    | `node.activation` (string)| Name like `'tanh'`, `'sigmoid'`, etc. |
| `aggregation`   | `node.aggregation` (string)| Name like `'sum'`, `'product'`, etc. |

Per-connection attributes:
- `weight`: float
- `enabled`: bool (disabled connections are excluded)

The CPU CTRNN computes each node's output as:

```
s_i = aggregation_i([w_j * u_j for j in inputs_i])
z_i = activation_i(bias_i + response_i * s_i)
du_i/dt = (-u_i + z_i) / tau_i
```

**Response parameter handling:** The `response` multiplier is applied *after*
aggregation and *before* the activation function. The weight matrix W stores
raw connection weights; `response` is a per-node scalar that scales the
aggregated input. This means two nodes receiving the same weighted sum can
produce different activations if they have different `response` values. The
GPU path must preserve this: `z = activation(bias + response * (W @ u))`,
not `z = activation(bias + W @ (response * u))`. The `response` array has
shape `[N, M]` and is applied elementwise after the matmul.

### Izhikevich (Spiking) Genomes

Each spiking genome uses `IZGenome` with `IZNodeGene` nodes. Per-node attributes:

| Attribute | Source     | Notes |
|-----------|------------|-------|
| `bias`    | `node.bias`| Constant current injection |
| `a`       | `node.a`   | Recovery time scale |
| `b`       | `node.b`   | Recovery sensitivity |
| `c`       | `node.c`   | Post-spike reset voltage |
| `d`       | `node.d`   | Post-spike recovery increment |

Connections are `DefaultConnectionGene` with `weight` and `enabled`.

### Node Identification

Both network types use `required_for_output()` to prune the genome down to
nodes that actually contribute to output. Only enabled connections to/from
required nodes are included in the phenotype.

Node keys follow the neat-python convention:
- Input pins: negative integers (-1, -2, ..., -num_inputs)
- Output nodes: 0, 1, ..., num_outputs-1
- Hidden nodes: integers >= num_outputs

## 2. Tensor Shapes and Index Mapping

### Node Index Mapping

For GPU evaluation, we map sparse node keys to dense indices within a
fixed-size array of `max_nodes` slots. The mapping is:

```
Index 0 .. num_inputs-1        → input pins (-1, -2, ..., -num_inputs)
Index num_inputs .. num_inputs+num_outputs-1  → output nodes (0, 1, ..., num_outputs-1)
Index num_inputs+num_outputs .. max_nodes-1   → hidden nodes (genome-specific)
```

Input and output node indices are the same for all genomes (since num_inputs
and num_outputs are fixed by config). Hidden node indices vary per genome —
each genome's hidden nodes are packed into the slots starting at
`num_inputs + num_outputs`, in arbitrary but consistent order.

The mapping is computed per genome during the CPU-side packing step.

### Tensor Dimensions

For a population of N genomes with at most M nodes (including inputs):

| Tensor          | Shape            | dtype    | Description |
|-----------------|------------------|----------|-------------|
| `W`             | `[N, M, M]`      | float32  | Weight matrices. W[g, i, j] = weight from node j to node i for genome g. Zero where no connection exists. |
| `node_mask`     | `[N, M]`         | bool     | True for active nodes in each genome. Always True for input/output slots. |
| `bias`          | `[N, M]`         | float32  | Per-node bias. Zero for inactive/input nodes. |
| `response`      | `[N, M]`         | float32  | Per-node response multiplier (CTRNN only). |
| `tau`           | `[N, M]`         | float32  | Per-node time constant (CTRNN only). |
| `activation_id` | `[N, M]`         | int32    | Per-node activation function index (CTRNN only). |
| `decay`         | `[N, M]`         | float32  | Precomputed exp(-dt/τ_i) (CTRNN only). |
| `scale`         | `[N, M]`         | float32  | Precomputed τ_i * (1 - decay_i) (CTRNN only). |
| `a, b, c, d`    | `[N, M]` each    | float32  | Izhikevich parameters (spiking only). |

State tensors during simulation:

| Tensor  | Shape       | dtype   | Description |
|---------|-------------|---------|-------------|
| `u`     | `[N, M]`    | float32 | Node state (CTRNN) or membrane voltage (spiking). |
| `recov` | `[N, M]`    | float32 | Recovery variable (spiking only). |
| `fired` | `[N, M]`    | float32 | Spike indicator, 0.0 or 1.0 (spiking only). |

### How max_nodes Is Determined

At each evaluation call, we iterate over the current population's genomes,
compute each genome's required node count (num_inputs + num_outputs +
num_hidden_required), and take the maximum. This value grows naturally as
NEAT complexifies networks. No fixed upper bound is assumed.

## 3. Input Signal Representation

The user provides an `input_fn` callable with the signature:

```python
def input_fn(t: float, dt: float) -> np.ndarray  # shape [num_inputs]
```

This function is called once per integration step on the CPU to produce the
input signal at time t. The returned array is broadcast across all N genomes
(same input for every genome in the population, which is the standard NEAT
evaluation pattern).

The full input trajectory is precomputed on CPU before GPU transfer:

```python
num_steps = int(t_max / dt)
inputs = np.zeros((num_steps, num_inputs), dtype=np.float32)
for step in range(num_steps):
    inputs[step] = input_fn(step * dt, dt)
```

This array is transferred to GPU once, shape `[num_steps, num_inputs]`.

At each integration step, the input values are written into the corresponding
columns of the state vector `u[:, 0:num_inputs]` for all genomes simultaneously.

**Alternative: per-genome input functions.** Some use cases may require different
inputs per genome (e.g., if the genome controls an agent in a simulation). For
this case, we also support:

```python
def input_fn(t: float, dt: float) -> np.ndarray  # shape [N, num_inputs]
```

The return shape is checked at the first call. If it returns `[num_inputs]`,
it is broadcast. If it returns `[N, num_inputs]`, it is used directly.

## 4. Output Trajectory Transfer

After GPU simulation completes, we need to get results back to CPU for the
user's fitness function.

### What to return

The evaluator collects the output node states at every integration step,
producing a trajectory tensor of shape `[N, num_steps, num_outputs]` on GPU.

This is transferred back to CPU as a NumPy array. The evaluator then calls
the user's fitness function once per genome:

```python
for i, (genome_id, genome) in enumerate(genomes):
    genome.fitness = fitness_fn(output_trajectory[i])  # shape [num_steps, num_outputs]
```

The output trajectory indices correspond to the output node slots:
`u[:, num_inputs:num_inputs+num_outputs]` at each time step.

### Memory considerations

For N=1000 genomes, T=1000 steps, O=2 outputs, float32:
- GPU memory: 1000 * 1000 * 2 * 4 bytes = 8 MB (negligible)
- CPU transfer: same 8 MB (negligible)

For larger simulations, we could optionally return only final states or
subsampled trajectories, but the full trajectory is small enough that this
optimization is unnecessary for typical use.

## 5. Activation and Aggregation Functions on GPU

### Aggregation

The GPU evaluator uses **sum aggregation only**. The batched matrix-vector
multiply `z = W @ u` computes `z_i = sum_j W_ij * u_j`, which is exactly
sum aggregation with weights. This is the natural aggregation for matrix
operations and by far the most common in NEAT configurations.

Genomes using non-sum aggregation (product, max, min, etc.) cannot be
evaluated on GPU. The evaluator will raise an error if any genome uses a
non-sum aggregation function.

### Activation Functions

The GPU evaluator supports the following activation functions via a CuPy
elementwise kernel that dispatches on `activation_id`:

| ID | Name       | Formula |
|----|------------|---------|
| 0  | `sigmoid`  | 1/(1+exp(-clip(5z, -60, 60))) |
| 1  | `tanh`     | tanh(clip(2.5z, -60, 60)) |
| 2  | `relu`     | max(0, z) |
| 3  | `identity` | z |
| 4  | `clamped`  | clip(z, -1, 1) |
| 5  | `elu`      | z if z>0 else exp(z)-1 |
| 6  | `softplus` | 0.2*log(1+exp(clip(5z, -60, 60))) |
| 7  | `sin`      | sin(clip(5z, -60, 60)) |
| 8  | `gauss`    | exp(-5*clip(z,-3.4,3.4)^2) |
| 9  | `abs`      | |z| |
| 10 | `square`   | z^2 |

These match the neat-python implementations exactly (same clipping, same
scaling constants). If a genome uses an unsupported activation, the evaluator
raises an error at packing time.

The elementwise kernel avoids branching overhead by using a switch/if-else
chain — acceptable because in practice most or all nodes in a population use
the same activation function, so the GPU threads follow the same branch.

## 6. Integration Methods

### CTRNN: Exponential Euler (ETD1)

The existing CPU CTRNN uses forward Euler:
```
u_i += (dt / tau_i) * (-u_i + z_i)
```

The GPU evaluator uses exponential Euler, which integrates the linear decay
term exactly:
```
decay_i = exp(-dt / tau_i)        # precomputed once
scale_i = tau_i * (1 - decay_i)   # precomputed once

# Each step:
z_i = activation_i(bias_i + response_i * sum_j(W_ij * u_j))
u_i = decay_i * u_i + scale_i * (z_i / tau_i + x_i)
```

Wait — let me be more careful with the derivation. The ODE is:

```
du_i/dt = (-u_i + z_i) / tau_i + x_i
```

where `z_i = activation(bias + response * W_i @ u)` and `x_i` is external input.

Treating `z_i + tau_i * x_i` as constant over one step (the "nonlinear" part),
the exact solution of `du/dt = -u/tau + f/tau` is:

```
u(t+h) = u(t) * exp(-h/tau) + f * (1 - exp(-h/tau))
```

where `f = z_i + tau_i * x_i`. So:

```
u_i(t+h) = decay_i * u_i(t) + (1 - decay_i) * (z_i + tau_i * x_i)
```

Or equivalently, precomputing `scale_i = 1 - decay_i`:

```
u_i(t+h) = decay_i * u_i(t) + scale_i * z_i + scale_i * tau_i * x_i
```

For the common case where `x_i = 0` for non-input nodes, this simplifies to:
```
u_i(t+h) = decay_i * u_i(t) + scale_i * z_i
```

For input nodes, the state is set directly from the input signal (no
integration needed), matching the CPU behavior where `ivalues[i] = v`.

**Per-step GPU operations:**
1. Set input node states: `u[:, 0:num_inputs] = inputs[step]`
2. Batched matrix-vector multiply: `s = W @ u`  → shape `[N, M]`
3. Compute activation: `z = activation(bias + response * s)`  (elementwise, dispatched by activation_id)
4. Update state: `u = decay * u + scale * z`  (elementwise)
5. Re-clamp input nodes: `u[:, 0:num_inputs] = inputs[step]`
6. Record output: `trajectory[:, step, :] = u[:, num_inputs:num_inputs+num_outputs]`

Steps 2-5 are pure CuPy array operations. No Python loops over nodes or genomes.

**Input node invariant:** Input nodes do not integrate — they are clamped to
the external signal. Step 1 sets them before the matmul so that downstream
nodes see the current input. Step 4's exponential Euler update would corrupt
them (applying decay/scale to a value that isn't a state variable), so step 5
overwrites them again. The `decay` and `scale` values for input node slots are
irrelevant and never used meaningfully; they can be set to any value during
packing (we use `decay=1, scale=0` for clarity, which would be a no-op if
step 5 didn't exist, but we re-clamp anyway to make the invariant explicit).

The `node_mask` is True for input and output slots in all genomes. It is only
False for hidden-node slots that are unused (padding). Rather than applying
`u *= node_mask` as a separate step, inactive hidden nodes are guaranteed zero
by construction: their W rows and columns are zero (no connections), their
bias is zero, and their initial state is zero. The mask is used during
*packing* validation, not during the integration loop.

### Spiking (Izhikevich): Fixed-Step Half-Step Method

The existing CPU implementation uses a half-step method for the voltage update
(two half-steps of the quadratic voltage equation per full step):

```
v += 0.5 * dt * (0.04*v^2 + 5*v + 140 - u + I)
v += 0.5 * dt * (0.04*v^2 + 5*v + 140 - u + I)
u += dt * a * (b*v - u)

if v > 30: v = c, u += d, fired = 1
```

The GPU evaluator uses the same method for numerical equivalence:

**Per-step GPU operations:**
1. Compute synaptic current: For non-input nodes, `I = bias + W @ fired_or_input`
   - For connections from input nodes: use input values directly
   - For connections from other neurons: use `fired` (0.0 or 1.0)
2. Half-step voltage: `v += 0.5 * dt * (0.04*v^2 + 5*v + 140 - u_recov + I)`
3. Half-step voltage: `v += 0.5 * dt * (0.04*v^2 + 5*v + 140 - u_recov + I)`
4. Update recovery: `u_recov += dt * a * (b*v - u_recov)`
5. Spike detection and reset:
   - `spiked = v > 30`
   - `v = where(spiked, c, v)`
   - `u_recov = where(spiked, u_recov + d, u_recov)`
   - `fired = where(spiked, 1.0, 0.0)`
6. Apply node mask: zero out inactive nodes
7. Record output: `trajectory[:, step, :] = fired[:, output_indices]`

Note: The existing CPU code clamps overflows by resetting to `c` without a
spike. On GPU, float32 overflow produces `inf`, which we handle:
`v = where(isinf(v), c, v)` with `fired = 0` for overflow cases.

## 7. Edge Cases

### Genome with no connections
All weight matrix entries are zero. The matrix-vector multiply produces zero.
CTRNN nodes decay toward `activation(bias + response * 0)`. Spiking neurons
receive only their bias current. This is correct behavior.

### Genome with only input-to-output connections
Hidden node slots are inactive (masked out). The weight matrix has nonzero
entries only in the rows corresponding to output nodes and columns
corresponding to input nodes. Works correctly with the general formulation.

### Self-connections
Allowed in recurrent networks. Represented as diagonal entries in W.
The matrix-vector multiply includes them naturally.

### Disabled connections
Excluded during packing. The weight matrix entry is zero.

### Very small or very large time constants (CTRNN)
- Small τ: `decay ≈ 0`, node responds nearly instantaneously. Numerically
  stable with exponential Euler (unlike forward Euler which requires dt < 2τ).
- Large τ: `decay ≈ 1`, node changes very slowly. No stability issue.
This is the key advantage of exponential Euler over the CPU's forward Euler.

### Population with highly variable topology
max_nodes is determined per generation. If one genome has 100 nodes and others
have 5, all are padded to 100. The overhead is in memory (100x100 vs 5x5 weight
matrices) but GPU parallelism across the population still provides a net win
for typical population sizes (100+). For extreme topology variance, the overhead
could become significant — we do not attempt to address this (it would require
bucketing genomes by size, which adds complexity for an unlikely scenario).

## 8. Spiking Network Input Convention

The existing CPU `IZNN` treats input nodes differently from internal neurons:
- `set_inputs()` stores input values
- During `advance()`, when a neuron's input comes from an input pin, it uses
  the stored input value directly; when it comes from another neuron, it uses
  that neuron's `fired` (0.0 or 1.0) value

For the GPU evaluator, we handle this by constructing the "source" vector
that gets multiplied by W at each step:

```python
source = fired.copy()
source[:, 0:num_inputs] = external_inputs[step]
I = bias + W @ source
```

This correctly routes external input values through input-node connections and
spike signals through neuron-to-neuron connections.

## 9. Module Structure

```
neat/gpu/
    __init__.py          # Lazy imports, _import_cupy() helper, capability detection
    _padding.py          # genome_to_tensors() — pure Python + NumPy, no CuPy
    _cupy_backend.py     # CuPy kernels and batch evaluation functions
    evaluator.py         # GPUCTRNNEvaluator, GPUIZNNEvaluator (public API)
```

### `_import_cupy()`

```python
def _import_cupy():
    try:
        import cupy
        return cupy
    except ImportError:
        raise ImportError(
            "CuPy is required for GPU evaluation but is not installed.\n"
            "Install it with: pip install 'neat-python[gpu]'\n"
            "Or install CuPy directly: pip install cupy-cuda12x"
        ) from None
```

Called lazily in `evaluator.py` when a GPU evaluator is instantiated, never
at `import neat` time.

### `_padding.py` Key Function

```python
def pack_ctrnn_population(genomes, config) -> dict:
    """
    Convert a list of (genome_id, genome) pairs into padded NumPy arrays.

    Returns dict with keys: W, bias, response, tau, node_mask, activation_id,
    output_indices, node_key_maps (list of {node_key: dense_index} per genome).
    """
```

And similarly `pack_iznn_population()` for spiking networks.

These functions use `required_for_output()` to determine active nodes, build
the key-to-index mapping, and fill the padded arrays. They are pure
Python/NumPy and independently testable.
