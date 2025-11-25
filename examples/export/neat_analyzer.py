#!/usr/bin/env python3
"""
NEAT Network Analyzer and Visualizer

Provides tools for analyzing and understanding NEAT network structure.
"""

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


class NEATAnalyzer:
    """Analyze NEAT network structure and properties."""
    
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.nodes = {n['id']: n for n in self.data['nodes']}
        self.connections = [c for c in self.data['connections'] if c['enabled']]
        self.topology = self.data['topology']
        
        # Build adjacency
        self.adjacency = defaultdict(list)
        self.reverse_adjacency = defaultdict(list)
        for conn in self.connections:
            self.adjacency[conn['from']].append(conn['to'])
            self.reverse_adjacency[conn['to']].append(conn['from'])
    
    def compute_network_depth(self) -> int:
        """
        Compute the maximum depth (longest path) from inputs to outputs.
        This indicates how many sequential computations are needed.
        """
        # BFS from inputs
        depths = {key: 0 for key in self.topology['input_keys']}
        queue = list(self.topology['input_keys'])
        
        while queue:
            node = queue.pop(0)
            current_depth = depths[node]
            
            for next_node in self.adjacency[node]:
                new_depth = current_depth + 1
                if next_node not in depths or new_depth > depths[next_node]:
                    depths[next_node] = new_depth
                    if next_node not in queue:
                        queue.append(next_node)
        
        max_depth = max(depths[key] for key in self.topology['output_keys'])
        return max_depth
    
    def compute_node_statistics(self) -> Dict:
        """Compute statistics about nodes."""
        stats = {
            'total_nodes': len(self.nodes),
            'input_nodes': len(self.topology['input_keys']),
            'output_nodes': len(self.topology['output_keys']),
            'hidden_nodes': len([n for n in self.nodes.values() if n['type'] == 'hidden']),
            'activation_functions': {},
            'bias_stats': {},
            'response_stats': {},
        }
        
        # Activation function distribution
        for node in self.nodes.values():
            act = node['activation']['name']
            stats['activation_functions'][act] = stats['activation_functions'].get(act, 0) + 1
        
        # Bias statistics (excluding inputs)
        biases = [n['bias'] for n in self.nodes.values() if n['type'] != 'input']
        if biases:
            stats['bias_stats'] = {
                'mean': np.mean(biases),
                'std': np.std(biases),
                'min': np.min(biases),
                'max': np.max(biases),
            }
        
        # Response statistics
        responses = [n['response'] for n in self.nodes.values() if n['type'] != 'input']
        if responses:
            stats['response_stats'] = {
                'mean': np.mean(responses),
                'std': np.std(responses),
                'all_ones': all(r == 1.0 for r in responses),
            }
        
        return stats
    
    def compute_connection_statistics(self) -> Dict:
        """Compute statistics about connections."""
        weights = [c['weight'] for c in self.connections]
        
        # Connection density
        max_connections = len(self.nodes) * (len(self.nodes) - 1)
        density = len(self.connections) / max_connections if max_connections > 0 else 0
        
        # In/out degree
        in_degrees = [len(self.reverse_adjacency[nid]) for nid in self.nodes.keys()]
        out_degrees = [len(self.adjacency[nid]) for nid in self.nodes.keys()]
        
        stats = {
            'total_connections': len(self.connections),
            'connection_density': density,
            'weight_stats': {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'positive': sum(1 for w in weights if w > 0),
                'negative': sum(1 for w in weights if w < 0),
            },
            'degree_stats': {
                'avg_in_degree': np.mean(in_degrees),
                'max_in_degree': np.max(in_degrees),
                'avg_out_degree': np.mean(out_degrees),
                'max_out_degree': np.max(out_degrees),
            }
        }
        
        return stats
    
    def find_critical_path(self) -> List[int]:
        """
        Find the critical path (longest path from input to output).
        This path determines minimum sequential computation time.
        """
        # Compute all paths from inputs to outputs
        def find_all_paths(start, end, path=[]):
            path = path + [start]
            if start == end:
                return [path]
            paths = []
            for node in self.adjacency[start]:
                if node not in path:  # avoid cycles
                    new_paths = find_all_paths(node, end, path)
                    paths.extend(new_paths)
            return paths
        
        # Find longest path
        longest = []
        for input_key in self.topology['input_keys']:
            for output_key in self.topology['output_keys']:
                paths = find_all_paths(input_key, output_key)
                for path in paths:
                    if len(path) > len(longest):
                        longest = path
        
        return longest
    
    def generate_graphviz(self) -> str:
        """
        Generate a Graphviz DOT format representation.
        Can be rendered with: dot -Tpng network.dot -o network.png
        """
        dot = ["digraph NEAT {"]
        dot.append("  rankdir=LR;")
        dot.append("  node [shape=circle];")
        dot.append("")
        
        # Define nodes with colors
        for node_id, node in self.nodes.items():
            if node['type'] == 'input':
                color = 'lightblue'
                shape = 'box'
            elif node['type'] == 'output':
                color = 'lightgreen'
                shape = 'box'
            else:
                color = 'lightyellow'
                shape = 'circle'
            
            label = f"{node_id}\\n{node['activation']['name']}"
            if node['type'] != 'input':
                label += f"\\nb={node['bias']:.2f}"
            
            dot.append(f'  {node_id} [label="{label}", fillcolor={color}, '
                      f'style=filled, shape={shape}];')
        
        dot.append("")
        
        # Define edges with weights
        for conn in self.connections:
            weight = conn['weight']
            color = 'red' if weight < 0 else 'blue'
            width = min(abs(weight), 5)
            dot.append(f'  {conn["from"]} -> {conn["to"]} '
                      f'[label="{weight:.2f}", color={color}, penwidth={width}];')
        
        dot.append("}")
        return "\n".join(dot)
    
    def print_detailed_report(self):
        """Print comprehensive network analysis."""
        print("=" * 80)
        print("NEAT NETWORK ANALYSIS")
        print("=" * 80)
        
        # Metadata
        meta = self.data.get('metadata', {})
        if meta:
            print("\nMetadata:")
            print(f"  Problem: {meta.get('problem', 'N/A')}")
            print(f"  Generation: {meta.get('generation', 'N/A')}")
            print(f"  Fitness: {meta.get('fitness', 'N/A')}")
            print(f"  Genome ID: {meta.get('genome_id', 'N/A')}")
        
        # Node statistics
        print("\nNode Statistics:")
        node_stats = self.compute_node_statistics()
        print(f"  Total nodes: {node_stats['total_nodes']}")
        print(f"  Input nodes: {node_stats['input_nodes']}")
        print(f"  Hidden nodes: {node_stats['hidden_nodes']}")
        print(f"  Output nodes: {node_stats['output_nodes']}")
        
        print("\n  Activation Functions:")
        for act, count in node_stats['activation_functions'].items():
            print(f"    {act}: {count}")
        
        if node_stats['bias_stats']:
            print("\n  Bias Statistics:")
            for key, val in node_stats['bias_stats'].items():
                print(f"    {key}: {val:.4f}")
        
        if node_stats['response_stats']:
            print("\n  Response Statistics:")
            for key, val in node_stats['response_stats'].items():
                if isinstance(val, bool):
                    print(f"    {key}: {val}")
                else:
                    print(f"    {key}: {val:.4f}")
        
        # Connection statistics
        print("\nConnection Statistics:")
        conn_stats = self.compute_connection_statistics()
        print(f"  Total connections: {conn_stats['total_connections']}")
        print(f"  Connection density: {conn_stats['connection_density']:.4f}")
        
        print("\n  Weight Statistics:")
        for key, val in conn_stats['weight_stats'].items():
            if isinstance(val, (int, float)):
                if isinstance(val, float):
                    print(f"    {key}: {val:.4f}")
                else:
                    print(f"    {key}: {val}")
        
        print("\n  Degree Statistics:")
        for key, val in conn_stats['degree_stats'].items():
            print(f"    {key}: {val:.2f}")
        
        # Network topology
        print("\nNetwork Topology:")
        depth = self.compute_network_depth()
        print(f"  Network depth: {depth}")
        
        critical_path = self.find_critical_path()
        if critical_path:
            print(f"  Critical path length: {len(critical_path)}")
            print(f"  Critical path: {' -> '.join(map(str, critical_path))}")
        
        # Complexity metrics
        print("\nComplexity Metrics:")
        param_count = len(self.connections) + sum(1 for n in self.nodes.values() if n['type'] != 'input')
        print(f"  Total parameters: {param_count}")
        print(f"    (weights: {len(self.connections)}, biases: {len([n for n in self.nodes.values() if n['type'] != 'input'])})")
        
        parallelism = node_stats['total_nodes'] / depth if depth > 0 else 0
        print(f"  Parallelism factor: {parallelism:.2f}")
        print(f"    (higher = more nodes can be computed in parallel)")
        
        print("\n" + "=" * 80)


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python neat_analyzer.py <network.json>")
        print("\nExample: python neat_analyzer.py xor_winner.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    # Create analyzer
    analyzer = NEATAnalyzer(json_path)
    
    # Print report
    analyzer.print_detailed_report()
    
    # Generate Graphviz visualization
    print("\nGenerating Graphviz visualization...")
    dot_content = analyzer.generate_graphviz()
    
    import os
    base_name = os.path.basename(json_path).replace('.json', '.dot')
    dot_filename = base_name
    with open(dot_filename, 'w') as f:
        f.write(dot_content)
    
    print(f"✓ Graphviz DOT file saved to: {dot_filename}")
    print("\nTo visualize, run:")
    print(f"  dot -Tpng {dot_filename} -o network.png")
    print("  (requires Graphviz: https://graphviz.org/)")


if __name__ == '__main__':
    main()
