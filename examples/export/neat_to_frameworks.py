#!/usr/bin/env python3
"""
Complete NEAT-Python to PyTorch/TensorFlow/ONNX Converter

This tool converts NEAT-Python exported neural networks to standard 
deep learning frameworks. It handles arbitrary topologies with any 
activation functions supported by NEAT.

Usage:
    python neat_to_frameworks.py network.json [--format pytorch|tensorflow|onnx|all]
    
Author: Alan (CodeReclaimers, LLC)
"""

import json
import argparse
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from pathlib import Path


class NEATNetwork:
    """
    Represents a NEAT network with methods to convert to various frameworks.
    
    The NEAT computation model:
    For each node (in topological order):
        1. Aggregate: sum all weighted inputs
        2. Scale: multiply by response parameter
        3. Bias: add bias term
        4. Activate: apply activation function
    
    Formula: output = activation(response * sum(weight_i * input_i) + bias)
    """
    
    def __init__(self, json_path: str):
        """Load NEAT network from JSON export."""
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.metadata = self.data.get('metadata', {})
        self.topology = self.data['topology']
        self.nodes = {node['id']: node for node in self.data['nodes']}
        self.connections = self.data['connections']
        
        # Build incoming connection map for efficient access
        self.incoming_connections = defaultdict(list)
        for conn in self.connections:
            if conn['enabled']:
                self.incoming_connections[conn['to']].append(conn)
        
        # Compute evaluation order
        self.evaluation_order = self._compute_topological_order()
        
        # Extract parameters for easy access
        self.weights_dict = {
            (c['from'], c['to']): c['weight'] 
            for c in self.connections if c['enabled']
        }
        self.biases_dict = {
            nid: node['bias'] 
            for nid, node in self.nodes.items() 
            if node['type'] != 'input'
        }
        self.responses_dict = {nid: node['response'] for nid, node in self.nodes.items()}
        self.activations_dict = {
            nid: node['activation']['name'] 
            for nid, node in self.nodes.items()
        }
    
    def _compute_topological_order(self) -> List[int]:
        """
        Compute evaluation order for nodes using topological sort.
        Ensures each node is evaluated only after all its inputs are ready.
        """
        # Start with input nodes (already computed)
        ready = set(self.topology['input_keys'])
        evaluation_order = []
        
        # Nodes that still need to be evaluated
        remaining = set(self.nodes.keys()) - ready
        
        while remaining:
            # Find nodes whose all inputs are ready
            newly_ready = [
                node_id for node_id in remaining
                if all(conn['from'] in ready for conn in self.incoming_connections[node_id])
            ]
            
            if not newly_ready:
                raise ValueError("Cycle detected in network topology")
            
            # Sort for deterministic ordering
            newly_ready.sort()
            
            for node_id in newly_ready:
                evaluation_order.append(node_id)
                ready.add(node_id)
                remaining.remove(node_id)
        
        return evaluation_order
    
    def _numpy_activation(self, name: str):
        """Get numpy activation function."""
        activations = {
            'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
            'tanh': np.tanh,
            'relu': lambda x: np.maximum(0, x),
            'identity': lambda x: x,
            'sin': np.sin,
            'cos': np.cos,
            'abs': np.abs,
            'square': np.square,
            'gauss': lambda x: np.exp(-x**2 / 2.0),
            'hat': lambda x: np.maximum(0, 1 - np.abs(x)),
        }
        return activations.get(name, lambda x: x)
    
    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate network using numpy.
        
        Args:
            inputs: Shape (batch_size, num_inputs) or (num_inputs,)
        
        Returns:
            outputs: Shape (batch_size, num_outputs) or (num_outputs,)
        """
        single_sample = inputs.ndim == 1
        if single_sample:
            inputs = inputs.reshape(1, -1)
        
        batch_size = inputs.shape[0]
        node_values = {}
        
        # Initialize input nodes
        for i, node_key in enumerate(self.topology['input_keys']):
            node_values[node_key] = inputs[:, i]
        
        # Evaluate nodes in topological order
        for node_id in self.evaluation_order:
            node = self.nodes[node_id]
            
            # Aggregate weighted inputs
            aggregated = np.zeros(batch_size, dtype=np.float32)
            for conn in self.incoming_connections[node_id]:
                weight = conn['weight']
                input_value = node_values[conn['from']]
                aggregated += weight * input_value
            
            # Apply response multiplier and bias
            aggregated = node['response'] * aggregated + node['bias']
            
            # Apply activation function
            activation_fn = self._numpy_activation(node['activation']['name'])
            node_values[node_id] = activation_fn(aggregated)
        
        # Extract output values
        output_values = [node_values[key] for key in self.topology['output_keys']]
        outputs = np.stack(output_values, axis=1)
        
        return outputs.squeeze(0) if single_sample else outputs
    
    def to_pytorch(self):
        """
        Convert to PyTorch nn.Module.
        
        Returns:
            torch.nn.Module: PyTorch model
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        neat_net = self
        
        class NEATModel(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Store network structure
                self.topology = neat_net.topology
                self.incoming = neat_net.incoming_connections
                self.eval_order = neat_net.evaluation_order
                self.node_info = neat_net.nodes
                
                # Create parameters for weights
                self.weights = nn.ParameterDict()
                for (from_id, to_id), weight in neat_net.weights_dict.items():
                    key = f"{from_id}_{to_id}"
                    self.weights[key] = nn.Parameter(
                        torch.tensor(weight, dtype=torch.float32),
                        requires_grad=False
                    )
                
                # Create parameters for biases and responses
                self.biases = nn.ParameterDict()
                self.responses = {}
                for node_id, bias in neat_net.biases_dict.items():
                    self.biases[str(node_id)] = nn.Parameter(
                        torch.tensor(bias, dtype=torch.float32),
                        requires_grad=False
                    )
                    self.responses[node_id] = neat_net.responses_dict[node_id]
            
            def forward(self, x):
                single = x.ndim == 1
                if single:
                    x = x.unsqueeze(0)
                
                node_values = {}
                
                # Set input values
                for i, key in enumerate(self.topology['input_keys']):
                    node_values[key] = x[:, i]
                
                # Evaluate nodes in order
                for node_id in self.eval_order:
                    node = self.node_info[node_id]
                    
                    # Aggregate inputs
                    agg = torch.zeros_like(x[:, 0])
                    for conn in self.incoming[node_id]:
                        key = f"{conn['from']}_{conn['to']}"
                        agg = agg + self.weights[key] * node_values[conn['from']]
                    
                    # Apply response and bias
                    response = self.responses[node_id]
                    agg = response * agg + self.biases[str(node_id)]
                    
                    # Apply activation
                    act_name = node['activation']['name']
                    if act_name == 'sigmoid':
                        node_values[node_id] = torch.sigmoid(agg)
                    elif act_name == 'tanh':
                        node_values[node_id] = torch.tanh(agg)
                    elif act_name == 'relu':
                        node_values[node_id] = torch.relu(agg)
                    elif act_name == 'sin':
                        node_values[node_id] = torch.sin(agg)
                    elif act_name == 'cos':
                        node_values[node_id] = torch.cos(agg)
                    elif act_name == 'abs':
                        node_values[node_id] = torch.abs(agg)
                    elif act_name == 'square':
                        node_values[node_id] = agg ** 2
                    elif act_name == 'gauss':
                        node_values[node_id] = torch.exp(-agg ** 2 / 2.0)
                    elif act_name == 'hat':
                        node_values[node_id] = torch.clamp(1 - torch.abs(agg), min=0)
                    else:  # identity
                        node_values[node_id] = agg
                
                # Extract outputs
                out = torch.stack([node_values[k] for k in self.topology['output_keys']], dim=1)
                return out.squeeze(0) if single else out
        
        return NEATModel()
    
    def to_tensorflow(self):
        """
        Convert to TensorFlow/Keras model using functional API.
        
        Returns:
            tf.keras.Model: TensorFlow model (already built)
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not installed. Install with: pip install tensorflow")
        
        neat_net = self
        
        class NEATLayer(tf.keras.layers.Layer):
            def __init__(self, neat_net: 'NEATNetwork', **kwargs):
                super().__init__(**kwargs)
                self.neat = neat_net
                
                # Create weight and bias variables
                self.weight_vars = {}
                self.bias_vars = {}
                
                for (from_id, to_id), weight in neat_net.weights_dict.items():
                    key = f"{from_id}_{to_id}"
                    self.weight_vars[key] = tf.Variable(
                        weight, dtype=tf.float32, trainable=False, name=f'w_{key}'
                    )
                
                for node_id, bias in neat_net.biases_dict.items():
                    self.bias_vars[str(node_id)] = tf.Variable(
                        bias, dtype=tf.float32, trainable=False, name=f'b_{node_id}'
                    )
            
            def call(self, inputs):
                node_values = {}
                
                # Set input values
                for i, key in enumerate(self.neat.topology['input_keys']):
                    node_values[key] = inputs[:, i]
                
                # Evaluate nodes
                for node_id in self.neat.evaluation_order:
                    node = self.neat.nodes[node_id]
                    
                    # Aggregate
                    agg = tf.zeros_like(inputs[:, 0])
                    for conn in self.neat.incoming_connections[node_id]:
                        key = f"{conn['from']}_{conn['to']}"
                        agg = agg + self.weight_vars[key] * node_values[conn['from']]
                    
                    # Response and bias
                    response = self.neat.responses_dict[node_id]
                    agg = response * agg + self.bias_vars[str(node_id)]
                    
                    # Activation
                    act_name = node['activation']['name']
                    if act_name == 'sigmoid':
                        node_values[node_id] = tf.sigmoid(agg)
                    elif act_name == 'tanh':
                        node_values[node_id] = tf.tanh(agg)
                    elif act_name == 'relu':
                        node_values[node_id] = tf.nn.relu(agg)
                    elif act_name == 'sin':
                        node_values[node_id] = tf.sin(agg)
                    elif act_name == 'cos':
                        node_values[node_id] = tf.cos(agg)
                    elif act_name == 'abs':
                        node_values[node_id] = tf.abs(agg)
                    elif act_name == 'square':
                        node_values[node_id] = agg ** 2
                    elif act_name == 'gauss':
                        node_values[node_id] = tf.exp(-agg ** 2 / 2.0)
                    elif act_name == 'hat':
                        node_values[node_id] = tf.maximum(0.0, 1 - tf.abs(agg))
                    else:  # identity
                        node_values[node_id] = agg
                
                return tf.stack([node_values[k] for k in self.neat.topology['output_keys']], axis=1)
        
        # Create functional model with explicit input layer
        num_inputs = len(self.topology['input_keys'])
        inputs = tf.keras.Input(shape=(num_inputs,), name='input')
        neat_layer = NEATLayer(self)
        outputs = neat_layer(inputs)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='NEAT_Network')
    
    def to_onnx(self, output_path: str, opset_version: int = None):
        """
        Export to ONNX format via PyTorch.
        
        Args:
            output_path: Where to save the ONNX file
            opset_version: ONNX opset version (default: None = use PyTorch's default)
                          None: Let PyTorch choose (cleanest, no warnings)
                          18+: Modern runtimes
                          13-17: Older ONNX Runtime compatibility
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for ONNX export. Install with: pip install torch")
        
        # Convert to PyTorch
        model = self.to_pytorch()
        model.eval()
        
        # Create dummy input
        num_inputs = len(self.topology['input_keys'])
        dummy_input = torch.randn(1, num_inputs)
        
        # Export to ONNX
        try:
            export_args = {
                'f': output_path,
                'export_params': True,
                'do_constant_folding': True,
                'input_names': ['input'],
                'output_names': ['output'],
                'dynamic_axes': {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            }
            
            # Only add opset_version if specified
            if opset_version is not None:
                export_args['opset_version'] = opset_version
            
            torch.onnx.export(model, dummy_input, **export_args)
            print(f"✓ ONNX model saved to {output_path}")
        except ImportError as e:
            if 'onnxscript' in str(e):
                raise ImportError(
                    "ONNX export requires onnxscript. Install with:\n"
                    "  pip install onnxscript\n"
                    "Or install both together:\n"
                    "  pip install torch onnx onnxscript"
                ) from e
            else:
                raise
    
    def save_pytorch(self, output_path: str):
        """Save PyTorch model state dict."""
        import torch
        model = self.to_pytorch()
        torch.save(model.state_dict(), output_path)
        print(f"✓ PyTorch weights saved to {output_path}")
    
    def save_tensorflow(self, output_dir: str, format: str = 'savedmodel'):
        """
        Save TensorFlow model.
        
        Args:
            output_dir: Output path
            format: 'savedmodel' (for deployment) or 'keras' (native format)
        """
        model = self.to_tensorflow()
        
        if format == 'savedmodel':
            # Use export() for SavedModel format (TF Serving, TFLite, etc.)
            try:
                model.export(output_dir)
                print(f"✓ TensorFlow SavedModel exported to {output_dir}/")
            except AttributeError:
                # Fallback for older TensorFlow versions
                model.save(output_dir)
                print(f"✓ TensorFlow model saved to {output_dir}/")
        elif format == 'keras':
            # Save in native Keras format
            keras_path = output_dir if output_dir.endswith('.keras') else f"{output_dir}.keras"
            model.save(keras_path)
            print(f"✓ TensorFlow model saved to {keras_path}")
        else:
            raise ValueError(f"Unknown format: {format}. Use 'savedmodel' or 'keras'")
    
    def print_summary(self):
        """Print network summary."""
        print("=" * 70)
        print("NEAT Network Summary")
        print("=" * 70)
        print(f"Problem:    {self.metadata.get('problem', 'N/A')}")
        print(f"Generation: {self.metadata.get('generation', 'N/A')}")
        print(f"Fitness:    {self.metadata.get('fitness', 'N/A'):.4f}")
        print(f"\nTopology:")
        print(f"  Inputs:  {len(self.topology['input_keys'])}")
        print(f"  Hidden:  {sum(1 for n in self.nodes.values() if n['type'] == 'hidden')}")
        print(f"  Outputs: {len(self.topology['output_keys'])}")
        print(f"  Total connections: {len([c for c in self.connections if c['enabled']])}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NEAT-Python networks to PyTorch/TensorFlow/ONNX"
    )
    parser.add_argument('input_json', help='NEAT network JSON file')
    parser.add_argument(
        '--format', 
        choices=['pytorch', 'tensorflow', 'onnx', 'all'],
        default='all',
        help='Output format (default: all)'
    )
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--test', action='store_true', help='Test network with sample inputs')
    
    args = parser.parse_args()
    
    # Load network
    print(f"Loading NEAT network from {args.input_json}...")
    net = NEATNetwork(args.input_json)
    net.print_summary()
    
    # Test if requested
    if args.test:
        print("\nTesting network...")
        test_input = np.random.randn(len(net.topology['input_keys']))
        output = net.evaluate(test_input)
        print(f"Sample input:  {test_input}")
        print(f"Sample output: {output}")
    
    # Convert
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(args.input_json).stem
    
    if args.format in ['pytorch', 'all']:
        try:
            pytorch_path = output_dir / f"{base_name}.pth"
            net.save_pytorch(str(pytorch_path))
        except ImportError as e:
            print(f"✗ PyTorch export skipped: {e}")
    
    if args.format in ['tensorflow', 'all']:
        try:
            tf_dir = output_dir / f"{base_name}_tf"
            net.save_tensorflow(str(tf_dir))
        except ImportError as e:
            print(f"✗ TensorFlow export skipped: {e}")
    
    if args.format in ['onnx', 'all']:
        try:
            onnx_path = output_dir / f"{base_name}.onnx"
            net.to_onnx(str(onnx_path))
        except ImportError as e:
            print(f"✗ ONNX export skipped: {e}")
            if 'onnxscript' in str(e):
                print("  Install with: pip install onnxscript")
    
    print("\nConversion complete!")


if __name__ == '__main__':
    main()
