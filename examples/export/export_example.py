"""
Example demonstrating JSON network export.

This script trains a simple XOR network and exports it to JSON format,
demonstrating how to use the export functionality for all network types.
"""

import os
import sys
import json

import neat
from neat.export import export_network_json

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


def run_export_demo():
    # Determine path to configuration file from XOR example
    local_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(local_dir)
    config_path = os.path.join(parent_dir, 'xor', 'config-feedforward')
    
    if not os.path.exists(config_path):
        print(f"Error: Could not find config file at {config_path}")
        print("Make sure you're running from the examples/export directory")
        sys.exit(1)
    
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Create population
    print("Training XOR network...")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    
    # Run for up to 50 generations (should be enough for XOR)
    winner = p.run(eval_genomes, 50)
    
    print('\n' + '='*70)
    print('Training complete!')
    print('='*70)
    print(f'\nBest genome fitness: {winner.fitness:.4f}')
    print(f'Genome ID: {winner.key}')
    print(f'Generation: {p.generation}')
    
    # Create network from winner
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Test the network
    print('\nTesting network:')
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print(f"  Input {xi}, expected {xo[0]:.1f}, got {output[0]:.4f}")
    
    # Export to JSON - Method 1: Return as string
    print('\n' + '='*70)
    print('Exporting network to JSON...')
    print('='*70)
    
    json_str = export_network_json(
        winner_net,
        metadata={
            'fitness': winner.fitness,
            'generation': p.generation,
            'genome_id': winner.key,
            'problem': 'XOR'
        }
    )
    
    # Parse and pretty-print to console
    data = json.loads(json_str)
    print(f"\nExported network summary:")
    print(f"  Format version: {data['format_version']}")
    print(f"  Network type: {data['network_type']}")
    print(f"  Inputs: {data['topology']['num_inputs']}")
    print(f"  Outputs: {data['topology']['num_outputs']}")
    print(f"  Nodes: {len(data['nodes'])}")
    print(f"  Connections: {len(data['connections'])}")
    print(f"  Fitness: {data['metadata'].get('fitness', 'N/A')}")
    
    # Export to JSON - Method 2: Write to file
    output_file = os.path.join(local_dir, 'xor_winner.json')
    export_network_json(
        winner_net,
        filepath=output_file,
        metadata={
            'fitness': winner.fitness,
            'generation': p.generation,
            'genome_id': winner.key,
            'problem': 'XOR',
            'note': 'Example export from neat-python'
        }
    )
    
    print(f"\nNetwork exported to: {output_file}")
    print("\nSample of exported JSON (first 20 lines):")
    print('-' * 70)
    
    # Display first part of the JSON
    lines = json_str.split('\n')
    for line in lines[:20]:
        print(line)
    if len(lines) > 20:
        print(f"... ({len(lines) - 20} more lines)")
    
    print('\n' + '='*70)
    print('Export complete!')
    print('='*70)
    print(f"\nYou can now:")
    print(f"  1. View the full export: cat {output_file}")
    print(f"  2. Validate JSON: python -m json.tool {output_file}")
    print(f"  3. Use a third-party tool to convert to ONNX, TensorFlow, etc.")
    print(f"\nSee docs/network-json-format.md for format documentation.")


if __name__ == '__main__':
    run_export_demo()
