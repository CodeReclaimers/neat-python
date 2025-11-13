"""
Edge case tests for graph algorithm implementations.

Tests cover complex scenarios and edge cases for:
- Cycle detection (creates_cycle)
- Required node identification (required_for_output)
- Feed-forward layer computation (feed_forward_layers)
"""

import unittest
from neat.graphs import creates_cycle, required_for_output, feed_forward_layers


class TestCycleDetection(unittest.TestCase):
    """Test cycle detection edge cases."""
    
    def test_self_loop(self):
        """Test that self-loops are detected as cycles."""
        # Self-loop should always create a cycle
        self.assertTrue(creates_cycle([], (5, 5)))
        self.assertTrue(creates_cycle([(0, 1)], (2, 2)))
    
    def test_direct_cycle(self):
        """Test detection of direct 2-node cycles."""
        # Adding reverse of existing edge creates cycle
        self.assertTrue(creates_cycle([(0, 1)], (1, 0)))
        self.assertTrue(creates_cycle([(5, 7)], (7, 5)))
    
    def test_multi_step_cycle(self):
        """Test detection of cycles with multiple intermediate nodes."""
        # 3-node cycle: 0 -> 1 -> 2 -> 0
        connections = [(0, 1), (1, 2)]
        self.assertTrue(creates_cycle(connections, (2, 0)))
        
        # 4-node cycle: 0 -> 1 -> 2 -> 3 -> 0
        connections = [(0, 1), (1, 2), (2, 3)]
        self.assertTrue(creates_cycle(connections, (3, 0)))
        
        # 5-node cycle
        connections = [(0, 1), (1, 2), (2, 3), (3, 4)]
        self.assertTrue(creates_cycle(connections, (4, 0)))
    
    def test_complex_network_with_cycle(self):
        """Test cycle detection in complex networks with branches."""
        # Network with multiple paths, adding edge creates cycle
        connections = [
            (0, 1), (0, 2),
            (1, 3), (2, 3),
            (3, 4)
        ]
        # Adding 4 -> 0 creates cycle through any path
        self.assertTrue(creates_cycle(connections, (4, 0)))
        # Adding 4 -> 1 creates cycle
        self.assertTrue(creates_cycle(connections, (4, 1)))
        # Adding 3 -> 0 creates cycle
        self.assertTrue(creates_cycle(connections, (3, 0)))
    
    def test_complex_network_no_cycle(self):
        """Test that valid feed-forward connections don't create cycles."""
        connections = [
            (0, 1), (0, 2),
            (1, 3), (2, 3),
            (3, 4)
        ]
        # Forward connections should not create cycles
        self.assertFalse(creates_cycle(connections, (0, 4)))
        self.assertFalse(creates_cycle(connections, (1, 4)))
        self.assertFalse(creates_cycle(connections, (2, 4)))
        self.assertFalse(creates_cycle(connections, (4, 5)))
    
    def test_disconnected_components(self):
        """Test cycle detection with disconnected graph components."""
        # Two separate chains
        connections = [(0, 1), (1, 2), (10, 11), (11, 12)]
        
        # Cycle within first component
        self.assertTrue(creates_cycle(connections, (2, 0)))
        # Cycle within second component
        self.assertTrue(creates_cycle(connections, (12, 10)))
        # No cycle connecting components
        self.assertFalse(creates_cycle(connections, (2, 10)))
        self.assertFalse(creates_cycle(connections, (12, 0)))
    
    def test_empty_graph(self):
        """Test cycle detection with no existing connections."""
        # Any non-self-loop edge in empty graph doesn't create cycle
        self.assertFalse(creates_cycle([], (0, 1)))
        self.assertFalse(creates_cycle([], (5, 10)))
        # Self-loop always creates cycle
        self.assertTrue(creates_cycle([], (0, 0)))
    
    def test_single_connection(self):
        """Test cycle detection with only one existing connection."""
        connections = [(0, 1)]
        
        # Self-loops create cycles
        self.assertTrue(creates_cycle(connections, (0, 0)))
        self.assertTrue(creates_cycle(connections, (1, 1)))
        
        # Reverse creates cycle
        self.assertTrue(creates_cycle(connections, (1, 0)))
        
        # Forward doesn't create cycle
        self.assertFalse(creates_cycle(connections, (1, 2)))
        self.assertFalse(creates_cycle(connections, (0, 2)))
    
    def test_diamond_pattern(self):
        """Test cycle detection in diamond-shaped network."""
        # Diamond: 0 -> 1,2 -> 3
        connections = [(0, 1), (0, 2), (1, 3), (2, 3)]
        
        # Closing the diamond creates cycle
        self.assertTrue(creates_cycle(connections, (3, 0)))
        self.assertTrue(creates_cycle(connections, (3, 1)))
        self.assertTrue(creates_cycle(connections, (3, 2)))
        
        # Forward edges don't create cycles
        self.assertFalse(creates_cycle(connections, (1, 2)))
        self.assertFalse(creates_cycle(connections, (2, 1)))  # Both are same layer
    
    def test_cycle_through_long_path(self):
        """Test cycle detection when cycle closes through long path."""
        # Long chain
        connections = [(i, i+1) for i in range(10)]
        
        # Closing from end to any earlier node creates cycle
        for i in range(10):
            self.assertTrue(creates_cycle(connections, (10, i)))
        
        # Forward connections don't create cycles
        self.assertFalse(creates_cycle(connections, (5, 11)))
        self.assertFalse(creates_cycle(connections, (10, 11)))


class TestRequiredForOutput(unittest.TestCase):
    """Test required_for_output edge cases."""
    
    def test_direct_connection(self):
        """Test simplest case: inputs directly connected to outputs."""
        inputs = [0, 1]
        outputs = [2]
        connections = [(0, 2), (1, 2)]
        
        required = required_for_output(inputs, outputs, connections)
        # Only output node is required (inputs excluded by convention)
        self.assertEqual(required, {2})
    
    def test_single_intermediate_layer(self):
        """Test with one hidden layer between inputs and outputs."""
        inputs = [0, 1]
        outputs = [5]
        connections = [(0, 2), (1, 3), (2, 5), (3, 5)]
        
        required = required_for_output(inputs, outputs, connections)
        # Hidden nodes 2, 3 and output 5 are required
        self.assertEqual(required, {2, 3, 5})
    
    def test_disconnected_nodes(self):
        """Test that disconnected nodes are not required."""
        inputs = [0, 1]
        outputs = [5]
        connections = [
            (0, 2), (2, 5),  # Path to output
            (1, 3), (3, 4)   # Disconnected from output
        ]
        
        required = required_for_output(inputs, outputs, connections)
        # Only nodes on path to output
        self.assertEqual(required, {2, 5})
    
    def test_redundant_connections(self):
        """Test network with multiple paths to same output."""
        inputs = [0, 1]
        outputs = [4]
        connections = [
            (0, 2), (1, 2),  # Both inputs to node 2
            (0, 3), (1, 3),  # Both inputs to node 3
            (2, 4), (3, 4)   # Both to output
        ]
        
        required = required_for_output(inputs, outputs, connections)
        # All nodes on paths to output
        self.assertEqual(required, {2, 3, 4})
    
    def test_multiple_outputs(self):
        """Test with multiple output nodes."""
        inputs = [0, 1]
        outputs = [5, 6]
        connections = [
            (0, 2), (2, 5),  # Path to output 5
            (1, 3), (3, 6),  # Path to output 6
            (2, 4)           # Dead-end path
        ]
        
        required = required_for_output(inputs, outputs, connections)
        # Nodes needed for both outputs, but not node 4
        self.assertEqual(required, {2, 3, 5, 6})
    
    def test_partially_connected_outputs(self):
        """Test when some outputs are unreachable from inputs."""
        inputs = [0, 1]
        outputs = [5, 6]
        connections = [
            (0, 2), (2, 5),  # Path to output 5
            (3, 6)           # Output 6 connected via orphaned node 3
        ]
        
        required = required_for_output(inputs, outputs, connections)
        # Node 3 is required even though it's orphaned (not reachable from inputs)
        # because it feeds into output 6. It acts as a "bias neuron".
        self.assertEqual(required, {2, 3, 5, 6})
    
    def test_recurrent_connections(self):
        """Test with recurrent (cyclic) connections."""
        inputs = [0, 1]
        outputs = [4]
        connections = [
            (0, 2), (1, 3),
            (2, 3), (3, 4),
            (4, 2)  # Recurrent connection
        ]
        
        required = required_for_output(inputs, outputs, connections)
        # All nodes in the cycle are required
        self.assertEqual(required, {2, 3, 4})
    
    def test_empty_connections(self):
        """Test with no connections."""
        inputs = [0, 1]
        outputs = [2]
        connections = []
        
        required = required_for_output(inputs, outputs, connections)
        # Only outputs (unreachable but still in output set)
        self.assertEqual(required, {2})
    
    def test_complex_branching_merging(self):
        """Test complex network with branching and merging paths."""
        inputs = [0, 1]
        outputs = [10]
        connections = [
            # First layer
            (0, 2), (0, 3), (1, 4), (1, 5),
            # Second layer
            (2, 6), (3, 6), (4, 7), (5, 7),
            # Third layer
            (6, 8), (7, 9),
            # Output layer
            (8, 10), (9, 10),
            # Dead branch
            (2, 11), (11, 12)
        ]
        
        required = required_for_output(inputs, outputs, connections)
        # All nodes on paths to output, excluding dead branch
        self.assertEqual(required, {2, 3, 4, 5, 6, 7, 8, 9, 10})
    
    def test_single_input_single_output(self):
        """Test minimal case with one input and one output."""
        inputs = [0]
        outputs = [1]
        connections = [(0, 1)]
        
        required = required_for_output(inputs, outputs, connections)
        self.assertEqual(required, {1})
    
    def test_unused_intermediate_nodes(self):
        """Test network where some intermediate nodes don't contribute to output."""
        inputs = [0, 1, 2]
        outputs = [8]
        connections = [
            (0, 3), (3, 8),     # Path 1 to output
            (1, 4), (4, 5),     # Dead path
            (2, 6), (6, 8)      # Path 2 to output
        ]
        
        required = required_for_output(inputs, outputs, connections)
        # Only nodes on paths to output
        self.assertEqual(required, {3, 6, 8})


class TestFeedForwardLayers(unittest.TestCase):
    """Test feed_forward_layers edge cases."""
    
    def test_single_layer(self):
        """Test simplest case: inputs directly to outputs."""
        inputs = [0, 1]
        outputs = [2]
        connections = [(0, 2), (1, 2)]
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        self.assertEqual(layers, [{2}])
        self.assertEqual(required, {2})
    
    def test_linear_chain(self):
        """Test linear chain of nodes."""
        inputs = [0]
        outputs = [4]
        connections = [(0, 1), (1, 2), (2, 3), (3, 4)]
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        self.assertEqual(layers, [{1}, {2}, {3}, {4}])
        self.assertEqual(required, {1, 2, 3, 4})
    
    def test_parallel_paths(self):
        """Test nodes that can be evaluated in parallel."""
        inputs = [0, 1]
        outputs = [4, 5]
        connections = [
            (0, 2), (2, 4),  # Path 1
            (1, 3), (3, 5)   # Path 2 (independent)
        ]
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        # Nodes 2 and 3 can be evaluated in parallel
        self.assertEqual(layers, [{2, 3}, {4, 5}])
        self.assertEqual(required, {2, 3, 4, 5})
    
    def test_diamond_structure(self):
        """Test diamond pattern: split then merge."""
        inputs = [0]
        outputs = [3]
        connections = [(0, 1), (0, 2), (1, 3), (2, 3)]
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        # Nodes 1 and 2 in parallel, then 3
        self.assertEqual(layers, [{1, 2}, {3}])
        self.assertEqual(required, {1, 2, 3})
    
    def test_dependent_layers(self):
        """Test nodes with dependencies must be in correct order."""
        inputs = [0]
        outputs = [4]
        connections = [
            (0, 1), (0, 2),
            (1, 3),
            (2, 3),
            (3, 4)
        ]
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        # Layer 1: 1, 2 (both need only input 0)
        # Layer 2: 3 (needs both 1 and 2)
        # Layer 3: 4 (needs 3)
        self.assertEqual(layers, [{1, 2}, {3}, {4}])
        self.assertEqual(required, {1, 2, 3, 4})
    
    def test_excludes_unused_nodes(self):
        """Test that unused nodes don't appear in layers."""
        inputs = [0, 1]
        outputs = [5]
        connections = [
            (0, 2), (2, 5),  # Used path
            (1, 3), (3, 4)   # Unused path
        ]
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        # Only nodes on path to output
        self.assertEqual(layers, [{2}, {5}])
        self.assertEqual(required, {2, 5})
    
    def test_complex_merge_points(self):
        """Test network with multiple merge points."""
        inputs = [0, 1]
        outputs = [7]
        connections = [
            (0, 2), (0, 3),
            (1, 2), (1, 3),
            (2, 4), (3, 5),
            (4, 6), (5, 6),
            (6, 7)
        ]
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        # Layer 1: 2, 3 (both need 0 and 1)
        # Layer 2: 4, 5 (need 2 and 3 respectively)
        # Layer 3: 6 (needs both 4 and 5)
        # Layer 4: 7 (needs 6)
        self.assertEqual(layers, [{2, 3}, {4, 5}, {6}, {7}])
        self.assertEqual(required, {2, 3, 4, 5, 6, 7})
    
    def test_wide_network(self):
        """Test network with many parallel nodes."""
        inputs = [0]
        outputs = [10]
        connections = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 10), (2, 10), (3, 10), (4, 10), (5, 10)
        ]
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        # All middle nodes in parallel, then output
        self.assertEqual(layers, [{1, 2, 3, 4, 5}, {10}])
        self.assertEqual(required, {1, 2, 3, 4, 5, 10})
    
    def test_deep_network(self):
        """Test network with many sequential layers."""
        inputs = [0]
        outputs = [5]
        connections = [(i, i+1) for i in range(5)]
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        # Each node in its own layer
        self.assertEqual(layers, [{1}, {2}, {3}, {4}, {5}])
        self.assertEqual(required, {1, 2, 3, 4, 5})
    
    def test_multiple_outputs_different_depths(self):
        """Test multiple outputs at different layer depths."""
        inputs = [0]
        outputs = [2, 4]
        connections = [
            (0, 1), (1, 2),  # Shallow path to output 2
            (2, 3), (3, 4)   # Deeper path to output 4
        ]
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        # Each layer contains nodes ready to evaluate
        self.assertEqual(layers, [{1}, {2}, {3}, {4}])
        self.assertEqual(required, {1, 2, 3, 4})
    
    def test_empty_network(self):
        """Test with no connections."""
        inputs = [0, 1]
        outputs = [2]
        connections = []
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        # Output node 2 is orphaned (no incoming connections), so it appears as first layer
        self.assertEqual(layers, [{2}])
        self.assertEqual(required, {2})
    
    def test_bottleneck_structure(self):
        """Test network with bottleneck (many-to-one-to-many)."""
        inputs = [0, 1, 2]
        outputs = [8, 9, 10]
        connections = [
            # Converge to bottleneck
            (0, 3), (1, 3), (2, 3),
            # Diverge from bottleneck
            (3, 4), (3, 5), (3, 6),
            # To outputs
            (4, 8), (5, 9), (6, 10)
        ]
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        # Layer 1: bottleneck
        # Layer 2: divergent nodes
        # Layer 3: outputs
        self.assertEqual(layers, [{3}, {4, 5, 6}, {8, 9, 10}])
        self.assertEqual(required, {3, 4, 5, 6, 8, 9, 10})
    
    def test_asymmetric_dependencies(self):
        """Test where nodes have different numbers of dependencies."""
        inputs = [0, 1, 2]
        outputs = [7]
        connections = [
            (0, 3),  # 3 depends only on 0
            (1, 4), (2, 4),  # 4 depends on 1 and 2
            (3, 5), (4, 5),  # 5 depends on 3 and 4
            (5, 6),
            (6, 7)
        ]
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        # Nodes 3 and 4 can be evaluated in parallel (both have all inputs ready)
        # Expected: [{3, 4}, {5}, {6}, {7}]
        self.assertEqual(len(layers), 4)  # 4 layers total
        self.assertEqual(layers[0], {3, 4})  # 3 and 4 can be evaluated in parallel
        self.assertEqual(layers[1], {5})  # 5 needs both 3 and 4
        self.assertEqual(layers[2], {6})  # 6 needs 5
        self.assertEqual(layers[3], {7})  # 7 needs 6
    
    def test_partial_input_usage(self):
        """Test where not all inputs are used."""
        inputs = [0, 1, 2]
        outputs = [5]
        connections = [
            (0, 3), (3, 5),  # Uses input 0
            (1, 4), (4, 5)   # Uses input 1
            # Input 2 not used
        ]
        
        layers, required = feed_forward_layers(inputs, outputs, connections)
        # Only nodes using connected inputs
        self.assertEqual(layers, [{3, 4}, {5}])
        self.assertEqual(required, {3, 4, 5})


class TestGraphAlgorithmsEdgeCases(unittest.TestCase):
    """Test edge cases across all graph algorithms."""
    
    def test_negative_node_ids(self):
        """Test that algorithms work with negative node IDs."""
        # Cycle detection
        self.assertTrue(creates_cycle([(-1, 0)], (0, -1)))
        self.assertFalse(creates_cycle([(-1, 0)], (0, 1)))
        
        # Required for output
        inputs = [-2, -1]
        outputs = [0]
        connections = [(-2, 0), (-1, 0)]
        required = required_for_output(inputs, outputs, connections)
        self.assertEqual(required, {0})
        
        # Feed forward layers
        layers, required = feed_forward_layers(inputs, outputs, connections)
        self.assertEqual(layers, [{0}])
    
    def test_large_node_ids(self):
        """Test with very large node ID numbers."""
        # Cycle detection
        connections = [(1000, 2000), (2000, 3000)]
        self.assertTrue(creates_cycle(connections, (3000, 1000)))
        
        # Required for output
        inputs = [1000, 1001]
        outputs = [3000]
        connections = [(1000, 2000), (1001, 2000), (2000, 3000)]
        required = required_for_output(inputs, outputs, connections)
        self.assertEqual(required, {2000, 3000})
    
    def test_sparse_node_ids(self):
        """Test with non-consecutive, sparse node IDs."""
        inputs = [1, 100]
        outputs = [500]
        connections = [
            (1, 50), (100, 200),
            (50, 300), (200, 300),
            (300, 500)
        ]
        
        required = required_for_output(inputs, outputs, connections)
        self.assertEqual(required, {50, 200, 300, 500})
        
        layers, req = feed_forward_layers(inputs, outputs, connections)
        self.assertEqual(len(layers), 3)
    
    def test_single_node_network(self):
        """Test with network containing only one node (output)."""
        inputs = [0]
        outputs = [1]
        connections = [(0, 1)]
        
        required = required_for_output(inputs, outputs, connections)
        self.assertEqual(required, {1})
        
        layers, req = feed_forward_layers(inputs, outputs, connections)
        self.assertEqual(layers, [{1}])
    
    def test_all_nodes_required(self):
        """Test network where every node contributes to output."""
        inputs = [0, 1]
        outputs = [6]
        connections = [
            (0, 2), (0, 3),
            (1, 2), (1, 3),
            (2, 4), (3, 5),
            (4, 6), (5, 6)
        ]
        
        required = required_for_output(inputs, outputs, connections)
        # All hidden nodes required
        self.assertEqual(required, {2, 3, 4, 5, 6})
        
        layers, req = feed_forward_layers(inputs, outputs, connections)
        # Verify all required nodes appear in layers
        all_layer_nodes = set()
        for layer in layers:
            all_layer_nodes.update(layer)
        self.assertEqual(all_layer_nodes, required)
    
    def test_duplicate_connections(self):
        """Test handling of duplicate connections in connection list."""
        inputs = [0]
        outputs = [2]
        # Same connection listed twice
        connections = [(0, 1), (1, 2), (0, 1)]
        
        # Should still work correctly
        required = required_for_output(inputs, outputs, connections)
        self.assertEqual(required, {1, 2})
        
        layers, req = feed_forward_layers(inputs, outputs, connections)
        self.assertEqual(layers, [{1}, {2}])


if __name__ == '__main__':
    unittest.main()
