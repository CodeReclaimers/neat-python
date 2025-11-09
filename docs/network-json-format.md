# NEAT Network JSON Export Format

## Overview

The NEAT-Python library exports trained neural networks to a JSON format that is:

- **Framework-agnostic**: Can be converted to ONNX, TensorFlow, PyTorch, CoreML, etc.
- **Human-readable**: Easy to inspect, debug, and understand
- **Version-controlled**: Includes format version for future compatibility
- **Complete**: Faithfully represents all network types and configurations

This document specifies the JSON format structure, provides examples, and offers guidance for creating converters to other formats.

## Format Version

**Current Version**: `1.0`

The format includes a `format_version` field to enable future evolution while maintaining backward compatibility. Third-party tools should check this version and handle accordingly.

## JSON Schema

### Top-Level Structure

```json
{
  "format_version": "1.0",
  "network_type": "feedforward | recurrent | ctrnn | iznn",
  "metadata": { ... },
  "topology": { ... },
  "nodes": [ ... ],
  "connections": [ ... ]
}
```

### Fields

#### `format_version` (string, required)

The version of this JSON format. Currently `"1.0"`.

#### `network_type` (string, required)

The type of neural network. Valid values:
- `"feedforward"` - Feed-forward network (no cycles)
- `"recurrent"` - Recurrent network (allows cycles)
- `"ctrnn"` - Continuous-Time Recurrent Neural Network
- `"iznn"` - Izhikevich Spiking Neural Network

#### `metadata` (object, required)

Contextual information about the network:

```json
{
  "created_timestamp": "2025-11-09T15:23:45Z",
  "neat_python_version": "1.0.0",
  "fitness": 98.5,
  "generation": 42,
  "genome_id": 123
}
```

Fields:
- `created_timestamp` (string, required): ISO 8601 timestamp
- `neat_python_version` (string, optional): Version of neat-python used
- `fitness` (number, optional): Fitness score of the genome
- `generation` (integer, optional): Generation number when exported
- `genome_id` (integer, optional): Unique genome identifier
- Additional custom fields may be included

#### `topology` (object, required)

Network input/output structure:

```json
{
  "num_inputs": 3,
  "num_outputs": 2,
  "input_keys": [-1, -2, -3],
  "output_keys": [0, 1]
}
```

Fields:
- `num_inputs` (integer): Number of input nodes
- `num_outputs` (integer): Number of output nodes
- `input_keys` (list of integers): Node IDs for inputs (typically negative)
- `output_keys` (list of integers): Node IDs for outputs (typically non-negative)

#### `nodes` (array, required)

List of nodes (neurons) in the network:

```json
{
  "id": 0,
  "type": "output",
  "activation": {
    "name": "sigmoid",
    "custom": false
  },
  "aggregation": {
    "name": "sum",
    "custom": false
  },
  "bias": 0.5,
  "response": 1.0
}
```

**Common fields:**
- `id` (integer): Unique node identifier
- `type` (string): One of `"input"`, `"hidden"`, `"output"`
- `activation` (object): Activation function information
  - `name` (string): Function name
  - `custom` (boolean): True if user-defined, false if built-in
- `aggregation` (object): Aggregation function information
  - `name` (string): Function name
  - `custom` (boolean): True if user-defined, false if built-in
- `bias` (number): Bias value
- `response` (number): Response multiplier

**CTRNN-specific fields:**
- `time_constant` (number): Time constant for continuous-time dynamics

**IZNN-specific fields:**
- `a` (number): Time scale of recovery variable
- `b` (number): Sensitivity of recovery variable
- `c` (number): After-spike reset value of membrane potential
- `d` (number): After-spike reset of recovery variable

#### `connections` (array, required)

List of weighted connections between nodes:

```json
{
  "from": -1,
  "to": 0,
  "weight": 0.75,
  "enabled": true
}
```

Fields:
- `from` (integer): Source node ID
- `to` (integer): Destination node ID
- `weight` (number): Connection weight
- `enabled` (boolean): Whether connection is active

## Built-in Functions Reference

### Activation Functions

All built-in activation functions are defined in `neat.activations`:

| Name | Formula | Notes |
|------|---------|-------|
| `sigmoid` | `1 / (1 + exp(-5z))` | Clamped to [-60, 60] |
| `tanh` | `tanh(2.5z)` | Clamped to [-60, 60] |
| `sin` | `sin(5z)` | Clamped to [-60, 60] |
| `gauss` | `exp(-5z²)` | Clamped to [-3.4, 3.4] |
| `relu` | `max(0, z)` | Rectified Linear Unit |
| `elu` | `z if z > 0 else exp(z) - 1` | Exponential Linear Unit |
| `lelu` | `z if z > 0 else 0.005z` | Leaky ReLU |
| `selu` | `λz if z > 0 else λα(exp(z) - 1)` | Scaled ELU (λ=1.0507, α=1.6732) |
| `softplus` | `0.2 * log(1 + exp(5z))` | Smooth approximation of ReLU |
| `identity` | `z` | Linear/pass-through |
| `clamped` | `clamp(z, -1, 1)` | Clamped to [-1, 1] |
| `inv` | `1/z` | Inverse (returns 0 on overflow) |
| `log` | `log(max(1e-7, z))` | Natural logarithm |
| `exp` | `exp(z)` | Exponential (clamped to [-60, 60]) |
| `abs` | `|z|` | Absolute value |
| `hat` | `max(0, 1 - |z|)` | Triangular/hat function |
| `square` | `z²` | Quadratic |
| `cube` | `z³` | Cubic |

### Aggregation Functions

All built-in aggregation functions are defined in `neat.aggregations`:

| Name | Description |
|------|-------------|
| `sum` | Sum of all inputs |
| `product` | Product of all inputs |
| `max` | Maximum value |
| `min` | Minimum value |
| `maxabs` | Value with maximum absolute value |
| `median` | Median value |
| `mean` | Arithmetic mean |

### Custom Functions

When a network uses custom (user-defined) activation or aggregation functions, the `custom` field is set to `true`. Third-party converters should handle these cases appropriately:

- **Error**: Refuse to convert if custom functions are unsupported
- **Warn**: Convert with a warning that behavior may not match
- **Approximate**: Map to closest built-in function in target framework

## Complete Examples

### Example 1: Simple Feedforward Network (XOR)

```json
{
  "format_version": "1.0",
  "network_type": "feedforward",
  "metadata": {
    "created_timestamp": "2025-11-09T15:30:00Z",
    "neat_python_version": "1.0.0",
    "fitness": 3.95,
    "generation": 150,
    "genome_id": 789
  },
  "topology": {
    "num_inputs": 2,
    "num_outputs": 1,
    "input_keys": [-1, -2],
    "output_keys": [0]
  },
  "nodes": [
    {
      "id": -1,
      "type": "input",
      "activation": {"name": "identity", "custom": false},
      "aggregation": {"name": "none", "custom": false},
      "bias": 0.0,
      "response": 1.0
    },
    {
      "id": -2,
      "type": "input",
      "activation": {"name": "identity", "custom": false},
      "aggregation": {"name": "none", "custom": false},
      "bias": 0.0,
      "response": 1.0
    },
    {
      "id": 0,
      "type": "output",
      "activation": {"name": "sigmoid", "custom": false},
      "aggregation": {"name": "sum", "custom": false},
      "bias": -0.123,
      "response": 1.0
    },
    {
      "id": 1,
      "type": "hidden",
      "activation": {"name": "relu", "custom": false},
      "aggregation": {"name": "sum", "custom": false},
      "bias": 0.456,
      "response": 1.0
    }
  ],
  "connections": [
    {"from": -1, "to": 1, "weight": 0.7, "enabled": true},
    {"from": -2, "to": 1, "weight": -0.5, "enabled": true},
    {"from": 1, "to": 0, "weight": 1.2, "enabled": true},
    {"from": -1, "to": 0, "weight": 0.3, "enabled": true}
  ]
}
```

### Example 2: Recurrent Network

```json
{
  "format_version": "1.0",
  "network_type": "recurrent",
  "metadata": {
    "created_timestamp": "2025-11-09T15:31:00Z",
    "neat_python_version": "1.0.0"
  },
  "topology": {
    "num_inputs": 1,
    "num_outputs": 1,
    "input_keys": [-1],
    "output_keys": [0]
  },
  "nodes": [
    {
      "id": -1,
      "type": "input",
      "activation": {"name": "identity", "custom": false},
      "aggregation": {"name": "none", "custom": false},
      "bias": 0.0,
      "response": 1.0
    },
    {
      "id": 0,
      "type": "output",
      "activation": {"name": "tanh", "custom": false},
      "aggregation": {"name": "sum", "custom": false},
      "bias": 0.0,
      "response": 1.0
    }
  ],
  "connections": [
    {"from": -1, "to": 0, "weight": 0.5, "enabled": true},
    {"from": 0, "to": 0, "weight": 0.8, "enabled": true}
  ]
}
```

Note the recurrent connection: `{"from": 0, "to": 0, ...}` creates a self-loop.

### Example 3: CTRNN

```json
{
  "format_version": "1.0",
  "network_type": "ctrnn",
  "metadata": {
    "created_timestamp": "2025-11-09T15:32:00Z",
    "neat_python_version": "1.0.0"
  },
  "topology": {
    "num_inputs": 2,
    "num_outputs": 1,
    "input_keys": [-1, -2],
    "output_keys": [0]
  },
  "nodes": [
    {
      "id": -1,
      "type": "input",
      "activation": {"name": "identity", "custom": false},
      "aggregation": {"name": "none", "custom": false},
      "bias": 0.0,
      "response": 1.0,
      "time_constant": 1.0
    },
    {
      "id": -2,
      "type": "input",
      "activation": {"name": "identity", "custom": false},
      "aggregation": {"name": "none", "custom": false},
      "bias": 0.0,
      "response": 1.0,
      "time_constant": 1.0
    },
    {
      "id": 0,
      "type": "output",
      "activation": {"name": "tanh", "custom": false},
      "aggregation": {"name": "sum", "custom": false},
      "bias": 0.1,
      "response": 1.0,
      "time_constant": 5.0
    }
  ],
  "connections": [
    {"from": -1, "to": 0, "weight": 0.6, "enabled": true},
    {"from": -2, "to": 0, "weight": 0.4, "enabled": true}
  ]
}
```

Note the `time_constant` field in CTRNN nodes for continuous-time dynamics.

### Example 4: IZNN (Izhikevich Spiking Network)

```json
{
  "format_version": "1.0",
  "network_type": "iznn",
  "metadata": {
    "created_timestamp": "2025-11-09T15:33:00Z",
    "neat_python_version": "1.0.0"
  },
  "topology": {
    "num_inputs": 1,
    "num_outputs": 1,
    "input_keys": [-1],
    "output_keys": [0]
  },
  "nodes": [
    {
      "id": -1,
      "type": "input",
      "activation": {"name": "identity", "custom": false},
      "aggregation": {"name": "none", "custom": false},
      "bias": 0.0,
      "response": 1.0,
      "a": 0.0,
      "b": 0.0,
      "c": 0.0,
      "d": 0.0
    },
    {
      "id": 0,
      "type": "output",
      "activation": {"name": "izhikevich", "custom": false},
      "aggregation": {"name": "sum", "custom": false},
      "bias": 5.0,
      "response": 1.0,
      "a": 0.02,
      "b": 0.20,
      "c": -65.0,
      "d": 8.0
    }
  ],
  "connections": [
    {"from": -1, "to": 0, "weight": 10.0, "enabled": true}
  ]
}
```

Note the Izhikevich model parameters (`a`, `b`, `c`, `d`) in IZNN nodes. These correspond to regular spiking behavior.

## Creating Converters to Other Formats

### General Approach

1. **Parse JSON**: Load and validate the JSON using the schema
2. **Map topology**: Create input/output nodes in target framework
3. **Map nodes**: Convert each node to target framework's equivalent
4. **Map connections**: Create weighted connections
5. **Map functions**: Convert activation/aggregation functions
6. **Export**: Save in target format

### Key Considerations

#### Activation Function Mapping

Many activation functions have direct equivalents in popular frameworks:

| NEAT Function | ONNX | TensorFlow/Keras | PyTorch |
|--------------|------|------------------|---------|
| `sigmoid` | `Sigmoid` | `sigmoid` | `torch.sigmoid` |
| `tanh` | `Tanh` | `tanh` | `torch.tanh` |
| `relu` | `Relu` | `relu` | `torch.relu` |
| `identity` | `Identity` | `linear` | `nn.Identity` |

For functions without direct equivalents (e.g., `gauss`, `hat`), you may need to:
- Compose from basic operations
- Use custom operators (framework-dependent)
- Approximate with similar functions (with warnings)

#### Aggregation Function Mapping

Most aggregations map straightforwardly:

| NEAT Aggregation | Operation |
|-----------------|-----------|
| `sum` | Element-wise sum |
| `product` | Element-wise product |
| `max`/`min` | Reduce-max/min operations |
| `mean` | Average across inputs |

#### Response Parameter

NEAT's `response` parameter scales the aggregated input before activation:

```
output = activation(bias + response * aggregation(inputs))
```

In target frameworks, this typically requires:
1. Aggregate inputs
2. Multiply by `response`
3. Add `bias`
4. Apply `activation`

#### Recurrent Networks

For recurrent networks:
- **ONNX**: Use LSTM/GRU operators or explicit recurrent connections
- **TensorFlow**: Use `tf.keras.layers.RNN` or custom cell
- **PyTorch**: Use `torch.nn.RNN` or manual state tracking

#### CTRNN

Continuous-time networks require differential equation solvers. Options:
- Discretize using Euler or Runge-Kutta methods
- Use framework's ODE solver (e.g., `torchdiffeq`)
- Approximate with standard RNN

#### IZNN

Spiking neural networks require specialized frameworks:
- **SpikingJelly** (PyTorch-based)
- **Norse** (PyTorch-based)
- **Brian2** (Python simulator)
- Custom implementation using the Izhikevich equations

### Validation

After conversion, validate that:
1. Network topology matches (same input/output dimensions)
2. Activation functions behave similarly
3. Forward pass produces similar results (within numerical tolerance)
4. Metadata is preserved or documented

## Versioning Strategy

The `format_version` field enables format evolution:

- **Minor changes** (backward-compatible): Add optional fields, new network types
- **Major changes** (breaking): Change required fields, remove fields, change semantics

When the format changes:
1. Increment `format_version`
2. Document changes in this file
3. Keep old versions supported in exporters for one major version
4. Provide migration tools if breaking changes occur

## FAQ

**Q: Can I add custom metadata fields?**  
A: Yes, add any additional fields to the `metadata` object.

**Q: How do I handle custom activation functions?**  
A: The `custom: true` flag indicates user-defined functions. Converters should either error, warn, or approximate.

**Q: Are weights and biases guaranteed to be exact?**  
A: Yes, all numeric values are exported as Python floats (IEEE 754 double precision).

**Q: Can I convert networks back to NEAT genomes?**  
A: Not currently. Export is one-way (phenotype only). Genome export may be added in a future version.

**Q: Which network types are supported?**  
A: All four types: FeedForwardNetwork, RecurrentNetwork, CTRNN, and IZNN.

**Q: Do I need neat-python to read the JSON?**  
A: No, the JSON format is self-contained and can be parsed by any JSON-compliant tool.

## Tools and Converters

While neat-python provides the JSON export capability, conversion to specific frameworks is left to third-party tools. This design:
- Keeps neat-python dependency-free
- Allows community-maintained converters
- Prevents "flavor of the year" bloat

If you create a converter tool, please consider:
- Open-sourcing it
- Documenting which format version you support
- Sharing it with the community

## Support

For questions about this format:
- GitHub Issues: https://github.com/CodeReclaimers/neat-python/issues
- Documentation: http://neat-python.readthedocs.io

For converter-specific questions, contact the converter's maintainer.
