"""
Network-specific export functions for each NEAT network type.

This module contains the implementation details for exporting each type of
NEAT network (FeedForward, Recurrent, CTRNN, IZNN) to the JSON format.
"""

from datetime import datetime
from .json_format import FORMAT_VERSION, get_function_info


def _get_neat_version():
    """Get the neat-python version if available."""
    try:
        import neat
        if hasattr(neat, '__version__'):
            return neat.__version__
        return "1.1.0"
    except:
        return "1.1.0"


def export_feedforward(network, metadata=None):
    """
    Export a FeedForwardNetwork to JSON-serializable dict.
    
    Args:
        network: neat.nn.FeedForwardNetwork instance
        metadata: Optional dict with fitness, generation, genome_id, etc.
    
    Returns:
        dict: JSON-serializable network representation
    """
    nodes = []
    connections = []
    
    # Extract node information from node_evals
    # node_evals format: (node, act_func, agg_func, bias, response, links)
    for node_id, act_func, agg_func, bias, response, links in network.node_evals:
        node_data = {
            "id": node_id,
            "type": "output" if node_id in network.output_nodes else "hidden",
            "activation": get_function_info(act_func, 'activation'),
            "aggregation": get_function_info(agg_func, 'aggregation'),
            "bias": bias,
            "response": response
        }
        nodes.append(node_data)
        
        # Extract connections for this node
        for input_id, weight in links:
            connections.append({
                "from": input_id,
                "to": node_id,
                "weight": weight,
                "enabled": True
            })
    
    # Add input nodes (they don't appear in node_evals but are part of topology)
    for input_id in network.input_nodes:
        nodes.append({
            "id": input_id,
            "type": "input",
            "activation": {"name": "identity", "custom": False},
            "aggregation": {"name": "none", "custom": False},
            "bias": 0.0,
            "response": 1.0
        })
    
    # Build the complete data structure
    data = {
        "format_version": FORMAT_VERSION,
        "network_type": "feedforward",
        "metadata": {
            "created_timestamp": datetime.utcnow().isoformat() + "Z",
            "neat_python_version": _get_neat_version(),
        },
        "topology": {
            "num_inputs": len(network.input_nodes),
            "num_outputs": len(network.output_nodes),
            "input_keys": network.input_nodes,
            "output_keys": network.output_nodes
        },
        "nodes": nodes,
        "connections": connections
    }
    
    # Add optional metadata if provided
    if metadata:
        data["metadata"].update(metadata)
    
    return data


def export_recurrent(network, metadata=None):
    """
    Export a RecurrentNetwork to JSON-serializable dict.
    
    Args:
        network: neat.nn.RecurrentNetwork instance
        metadata: Optional dict with fitness, generation, genome_id, etc.
    
    Returns:
        dict: JSON-serializable network representation
    """
    nodes = []
    connections = []
    
    # Extract node information from node_evals
    # node_evals format: (node, act_func, agg_func, bias, response, links)
    for node_id, act_func, agg_func, bias, response, links in network.node_evals:
        node_data = {
            "id": node_id,
            "type": "output" if node_id in network.output_nodes else "hidden",
            "activation": get_function_info(act_func, 'activation'),
            "aggregation": get_function_info(agg_func, 'aggregation'),
            "bias": bias,
            "response": response
        }
        nodes.append(node_data)
        
        # Extract connections for this node
        for input_id, weight in links:
            connections.append({
                "from": input_id,
                "to": node_id,
                "weight": weight,
                "enabled": True
            })
    
    # Add input nodes
    for input_id in network.input_nodes:
        nodes.append({
            "id": input_id,
            "type": "input",
            "activation": {"name": "identity", "custom": False},
            "aggregation": {"name": "none", "custom": False},
            "bias": 0.0,
            "response": 1.0
        })
    
    # Build the complete data structure
    data = {
        "format_version": FORMAT_VERSION,
        "network_type": "recurrent",
        "metadata": {
            "created_timestamp": datetime.utcnow().isoformat() + "Z",
            "neat_python_version": _get_neat_version(),
        },
        "topology": {
            "num_inputs": len(network.input_nodes),
            "num_outputs": len(network.output_nodes),
            "input_keys": network.input_nodes,
            "output_keys": network.output_nodes
        },
        "nodes": nodes,
        "connections": connections
    }
    
    # Add optional metadata if provided
    if metadata:
        data["metadata"].update(metadata)
    
    return data


def export_ctrnn(network, metadata=None):
    """
    Export a CTRNN (Continuous-Time Recurrent Neural Network) to JSON-serializable dict.
    
    Args:
        network: neat.ctrnn.CTRNN instance
        metadata: Optional dict with fitness, generation, genome_id, etc.
    
    Returns:
        dict: JSON-serializable network representation
    """
    nodes = []
    connections = []
    
    # Extract node information from node_evals dict
    # node_evals format: {node_id: CTRNNNodeEval(time_constant, activation, aggregation, bias, response, links)}
    for node_id, node_eval in network.node_evals.items():
        node_data = {
            "id": node_id,
            "type": "output" if node_id in network.output_nodes else "hidden",
            "activation": get_function_info(node_eval.activation, 'activation'),
            "aggregation": get_function_info(node_eval.aggregation, 'aggregation'),
            "bias": node_eval.bias,
            "response": node_eval.response,
            "time_constant": node_eval.time_constant
        }
        nodes.append(node_data)
        
        # Extract connections for this node
        for input_id, weight in node_eval.links:
            connections.append({
                "from": input_id,
                "to": node_id,
                "weight": weight,
                "enabled": True
            })
    
    # Add input nodes
    for input_id in network.input_nodes:
        nodes.append({
            "id": input_id,
            "type": "input",
            "activation": {"name": "identity", "custom": False},
            "aggregation": {"name": "none", "custom": False},
            "bias": 0.0,
            "response": 1.0,
            "time_constant": 1.0
        })
    
    # Build the complete data structure
    data = {
        "format_version": FORMAT_VERSION,
        "network_type": "ctrnn",
        "metadata": {
            "created_timestamp": datetime.utcnow().isoformat() + "Z",
            "neat_python_version": _get_neat_version(),
        },
        "topology": {
            "num_inputs": len(network.input_nodes),
            "num_outputs": len(network.output_nodes),
            "input_keys": network.input_nodes,
            "output_keys": network.output_nodes
        },
        "nodes": nodes,
        "connections": connections
    }
    
    # Add optional metadata if provided
    if metadata:
        data["metadata"].update(metadata)
    
    return data


def export_iznn(network, metadata=None):
    """
    Export an IZNN (Izhikevich Spiking Neural Network) to JSON-serializable dict.
    
    Args:
        network: neat.iznn.IZNN instance
        metadata: Optional dict with fitness, generation, genome_id, etc.
    
    Returns:
        dict: JSON-serializable network representation
    """
    nodes = []
    connections = []
    
    # Extract node information from neurons dict
    # neurons format: {node_id: IZNeuron(bias, a, b, c, d, inputs)}
    for node_id, neuron in network.neurons.items():
        node_data = {
            "id": node_id,
            "type": "output" if node_id in network.outputs else "hidden",
            "activation": {"name": "izhikevich", "custom": False},
            "aggregation": {"name": "sum", "custom": False},
            "bias": neuron.bias,
            "response": 1.0,
            "a": neuron.a,
            "b": neuron.b,
            "c": neuron.c,
            "d": neuron.d
        }
        nodes.append(node_data)
        
        # Extract connections for this neuron
        for input_id, weight in neuron.inputs:
            connections.append({
                "from": input_id,
                "to": node_id,
                "weight": weight,
                "enabled": True
            })
    
    # Add input nodes
    for input_id in network.inputs:
        nodes.append({
            "id": input_id,
            "type": "input",
            "activation": {"name": "identity", "custom": False},
            "aggregation": {"name": "none", "custom": False},
            "bias": 0.0,
            "response": 1.0,
            "a": 0.0,
            "b": 0.0,
            "c": 0.0,
            "d": 0.0
        })
    
    # Build the complete data structure
    data = {
        "format_version": FORMAT_VERSION,
        "network_type": "iznn",
        "metadata": {
            "created_timestamp": datetime.utcnow().isoformat() + "Z",
            "neat_python_version": _get_neat_version(),
        },
        "topology": {
            "num_inputs": len(network.inputs),
            "num_outputs": len(network.outputs),
            "input_keys": network.inputs,
            "output_keys": network.outputs
        },
        "nodes": nodes,
        "connections": connections
    }
    
    # Add optional metadata if provided
    if metadata:
        data["metadata"].update(metadata)
    
    return data
