# NEAT-Python Network Export and Analysis Tools

This directory contains tools for exporting, analyzing, and converting NEAT-Python neural networks to other frameworks.

NOTE: These conversion tools are a work in progress.  If you find they do not work correctly please open a GitHub issue.

## Overview

The export functionality allows you to:
1. Train a NEAT network and export it to JSON format
2. Analyze the exported network structure and properties
3. Convert the network to PyTorch, TensorFlow, or ONNX formats

## Usage Workflow

1. First, run `export_example.py` to train a simple XOR network and export it to JSON:
   ```bash
   python export_example.py
   ```
   This will create a `xor_winner.json` file containing the exported network.

2. Next, use `neat_analyzer.py` to analyze the exported network:
   ```bash
   python neat_analyzer.py xor_winner.json
   ```
   This will provide detailed statistics about the network structure and generate a Graphviz visualization.

3. Finally, use `neat_to_frameworks.py` to convert the network to other formats:
   ```bash
   python neat_to_frameworks.py xor_winner.json --format all
   ```
   This will create PyTorch, TensorFlow, and ONNX versions of the network.

## Converter Features

Convert NEAT-Python exported neural networks to PyTorch, TensorFlow, and ONNX formats.

## Converter Features

- **Complete topology preservation**: Handles arbitrary network structures, not just layered architectures
- **All activation functions**: Supports sigmoid, tanh, relu, identity, sin, cos, abs, square, gauss, hat
- **Multiple targets**: Export to PyTorch, TensorFlow, or ONNX
- **Verified correctness**: Numpy reference implementation for validation

## Installing dependencies

```bash
# For PyTorch support
pip install torch

# For TensorFlow support  
pip install tensorflow

# For ONNX (requires PyTorch + onnxscript)
pip install torch onnxscript

# Or install everything at once
pip install torch tensorflow onnxscript
```

## Usage

### Command Line

```bash
# Convert to all formats
python neat_to_frameworks.py xor_winner.json --format all --output-dir ./models

# Convert to specific format
python neat_to_frameworks.py xor_winner.json --format pytorch
python neat_to_frameworks.py xor_winner.json --format tensorflow
python neat_to_frameworks.py xor_winner.json --format onnx

# Test the network
python neat_to_frameworks.py xor_winner.json --test
```

### Python API

```python
from neat_to_frameworks import NEATNetwork

# Load network (created by export_example.py)
net = NEATNetwork('xor_winner.json')

# Test with numpy (always available)
import numpy as np
inputs = np.array([[0.0, 1.0]])
outputs = net.evaluate(inputs)
print(outputs)

# Convert to PyTorch
pytorch_model = net.to_pytorch()
import torch
output = pytorch_model(torch.tensor([[0.0, 1.0]], dtype=torch.float32))

# Convert to TensorFlow
tf_model = net.to_tensorflow()
import tensorflow as tf
output = tf_model(tf.constant([[0.0, 1.0]], dtype=tf.float32))

# Export to ONNX
net.to_onnx('model.onnx')  # Uses PyTorch's default opset (cleanest, no warnings)

# Or specify opset version for specific compatibility needs:
# net.to_onnx('model.onnx', opset_version=17)  # ONNX Runtime 1.13+
# net.to_onnx('model.onnx', opset_version=13)  # ONNX Runtime 1.10+

# Save models
net.save_pytorch('model.pth')
net.save_tensorflow('model_tf')  # SavedModel format (default)
# Or: net.save_tensorflow('model', format='keras')  # Keras format
```

## ONNX Opset Versions

The converter defaults to `None` (let PyTorch choose), which produces the cleanest export with no version conversion warnings.

| Opset | Use Case |
|-------|----------|
| None (default) | **Recommended** - PyTorch chooses best version, no warnings |
| 18-19 | Modern ONNX Runtime (1.14+), most runtimes support this |
| 17 | ONNX Runtime 1.13+ |
| 13-15 | Older ONNX Runtime (1.10+) |
| 11 | Legacy systems only |

**Note:** Most modern ONNX runtimes (2022+) support opset 18+. Only specify an older opset if deploying to legacy systems.

## Network Computation Model

NEAT networks compute each node as follows:

```
For each node (in topological order):
    1. Aggregate: weighted_sum = Σ(weight_i × input_i)
    2. Scale:     scaled = response × weighted_sum
    3. Bias:      biased = scaled + bias
    4. Activate:  output = activation(biased)
```

This differs from standard neural networks in a few ways:
- **Arbitrary topology**: Not restricted to layers
- **Response multiplier**: Additional scaling parameter (usually 1.0)
- **Topological evaluation**: Nodes computed in dependency order, not layer order

## TensorFlow Model Formats

The converter supports two TensorFlow save formats:

1. **SavedModel format** (default): `net.save_tensorflow('model_tf')`
   - Best for deployment (TF Serving, TFLite, TensorFlow.js)
   - Creates a directory with saved model
   - Load with: `tf.saved_model.load('model_tf')`

2. **Keras format**: `net.save_tensorflow('model', format='keras')`
   - Native Keras format (.keras file)
   - More compact single file
   - Load with: `tf.keras.models.load_model('model.keras')`

## Key Differences from Standard Networks

| Aspect | Standard NN | NEAT |
|--------|-------------|------|
| Structure | Layered | Arbitrary DAG |
| Weights | Dense matrices | Sparse connections |
| Evaluation | Layer-by-layer | Topological order |
| Node params | Bias only | Bias + response |

## Example: XOR Network

The `export_example.py` script will generate a XOR network with:
- 2 input nodes (keys: -1, -2)
- 2 hidden nodes (keys: variable)
- 1 output node (key: 0)
- Several enabled connections

The exact structure will vary depending on the NEAT evolution, but the evaluation order will always follow the topological sorting of the network nodes.

## Network Analysis

The `neat_analyzer.py` script provides detailed analysis of exported networks, including:
- Node and connection statistics
- Network depth and critical path analysis
- Complexity metrics
- Graphviz visualization generation

To analyze a network:
```bash
python neat_analyzer.py xor_winner.json
```

This will generate a detailed report and a Graphviz DOT file that can be converted to an image:
```bash
dot -Tpng xor_winner.dot -o network.png
```

## Extending the Converter

To add custom activation functions:

1. Add to `_numpy_activation()` method
2. Add corresponding PyTorch operation in `to_pytorch()`
3. Add corresponding TensorFlow operation in `to_tensorflow()`

Example:
```python
# In _numpy_activation()
'my_activation': lambda x: np.custom_function(x)

# In PyTorch forward()
elif act_name == 'my_activation':
    node_values[node_id] = custom_torch_function(agg)

# In TensorFlow call()
elif act_name == 'my_activation':
    node_values[node_id] = custom_tf_function(agg)
```

## Limitations

1. **Custom activations**: If your NEAT config uses custom activation functions, you'll need to implement them in PyTorch/TensorFlow
2. **Network size**: Very large networks may be slow to evaluate due to sequential node computation
3. **Recurrent networks**: This converter handles feedforward networks only (no recurrent connections)

## Troubleshooting

**Q: Getting "Cycle detected" error?**  
A: Your network has recurrent connections. NEAT-Python's feedforward mode should prevent this, but check your configuration.

**Q: Different outputs from NEAT-Python vs converted model?**  
A: Verify:
- Same input preprocessing
- Same activation functions
- No custom aggregation functions
- Response values are all 1.0 (or correctly handled)

**Q: ImportError for torch/tensorflow?**  
A: Install the required framework:
```bash
pip install torch  # for PyTorch/ONNX
pip install tensorflow  # for TensorFlow
```

**Q: FileNotFoundError for xor_winner.json?**
A: Make sure you've run `python export_example.py` first to generate the exported network file.