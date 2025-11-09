"""
Network export functionality for neat-python.

This module provides functions to export NEAT networks to JSON format.
The JSON format is framework-agnostic and human-readable, designed to enable
third-party tools to convert networks to various formats (ONNX, TensorFlow, PyTorch, etc.).

Example usage:
    import neat
    from neat.export import export_network_json
    
    # After training...
    winner = population.run(eval_genomes, 300)
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Export to JSON string
    json_str = export_network_json(net, metadata={'fitness': winner.fitness})
    
    # Or export directly to file
    export_network_json(net, filepath='network.json', metadata={'fitness': winner.fitness})
"""

import json
from .exporters import export_feedforward, export_recurrent, export_ctrnn, export_iznn
from .json_format import validate_json


def export_network_json(network, filepath=None, metadata=None):
    """
    Export a NEAT network to JSON format.
    
    This function supports all NEAT network types:
    - FeedForwardNetwork
    - RecurrentNetwork
    - CTRNN (Continuous-Time Recurrent Neural Network)
    - IZNN (Izhikevich Spiking Neural Network)
    
    The exported JSON format is well-documented and designed to be converted
    to other formats (ONNX, TensorFlow, PyTorch, etc.) by third-party tools.
    See docs/network-json-format.md for complete format specification.
    
    Args:
        network: A NEAT network instance (FeedForwardNetwork, RecurrentNetwork, CTRNN, or IZNN)
        filepath: Optional path to write JSON file. If None, returns JSON string.
        metadata: Optional dict with additional information to include in export.
                  Common fields: 'fitness', 'generation', 'genome_id'
    
    Returns:
        str: JSON string representation of the network (always returned)
    
    Raises:
        ValueError: If network type is not supported
        TypeError: If network is not a valid NEAT network instance
    
    Example:
        >>> import neat
        >>> from neat.export import export_network_json
        >>> 
        >>> # Create a network
        >>> config = neat.Config(...)
        >>> genome = ...
        >>> net = neat.nn.FeedForwardNetwork.create(genome, config)
        >>> 
        >>> # Export to string
        >>> json_str = export_network_json(net)
        >>> 
        >>> # Export to file with metadata
        >>> export_network_json(
        ...     net,
        ...     filepath='my_network.json',
        ...     metadata={
        ...         'fitness': 98.5,
        ...         'generation': 42,
        ...         'genome_id': 123
        ...     }
        ... )
    """
    if network is None:
        raise TypeError("network cannot be None")
    
    # Determine network type and call appropriate exporter
    network_type_name = type(network).__name__
    
    if 'FeedForward' in network_type_name:
        data = export_feedforward(network, metadata)
    elif 'Recurrent' in network_type_name and 'CTRNN' not in network_type_name:
        data = export_recurrent(network, metadata)
    elif 'CTRNN' in network_type_name:
        data = export_ctrnn(network, metadata)
    elif 'IZNN' in network_type_name:
        data = export_iznn(network, metadata)
    else:
        raise ValueError(
            f"Unsupported network type: {network_type_name}. "
            f"Supported types: FeedForwardNetwork, RecurrentNetwork, CTRNN, IZNN"
        )
    
    # Validate the generated JSON structure
    validate_json(data)
    
    # Convert to JSON string
    json_str = json.dumps(data, indent=2)
    
    # Write to file if filepath provided
    if filepath is not None:
        with open(filepath, 'w') as f:
            f.write(json_str)
    
    return json_str


# Export public API
__all__ = ['export_network_json']
