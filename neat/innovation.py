"""
Innovation tracking for NEAT.

This module implements the innovation numbering system described in:
Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through
augmenting topologies. Evolutionary computation, 10(2), 99-127.

From the paper (p. 108):
"Whenever a new gene appears (through structural mutation), a global innovation
number is incremented and assigned to that gene."

"By keeping a list of the innovations that occurred in the current generation,
it is possible to ensure that when the same structure arises more than once
through independent mutations in the same generation, each identical mutation
is assigned the same innovation number."
"""


class InnovationTracker:
    """
    Tracks innovation numbers for structural mutations in NEAT.
    
    This class maintains:
    1. A global counter that increments across all generations
    2. A generation-specific dictionary for deduplication of identical mutations
       within the same generation
    
    Innovation numbers are assigned when:
    - A new connection is added between two nodes
    - A connection is split by adding a node (creates two new connections)
    - Initial connections are created in the starting population
    
    The tracker ensures that if multiple genomes independently make the same
    structural mutation in one generation, they receive the same innovation number.
    This enables proper gene alignment during crossover.
    """
    
    def __init__(self, start_number=0):
        """
        Initialize the innovation tracker.
        
        Args:
            start_number: The initial value for the global counter (default: 0).
                         The first innovation will be start_number + 1.
        """
        self.global_counter = start_number
        # Maps (input_node, output_node, mutation_type) -> innovation_number
        # This is cleared at the start of each generation
        self.generation_innovations = {}
    
    def get_innovation_number(self, input_node, output_node, mutation_type='add_connection'):
        """
        Get or assign an innovation number for a structural mutation.
        
        If this exact mutation (same nodes and type) has already occurred in the
        current generation, returns the existing innovation number. Otherwise,
        increments the global counter and assigns a new innovation number.
        
        Args:
            input_node: The input node ID for the connection
            output_node: The output node ID for the connection
            mutation_type: Type of mutation:
                - 'add_connection': A new connection was added
                - 'add_node_in': Connection from original input to new node
                - 'add_node_out': Connection from new node to original output
                - 'initial_connection': Connection in initial population
        
        Returns:
            int: The innovation number for this structural mutation
        
        Example:
            >>> tracker = InnovationTracker()
            >>> # First genome adds connection 1->2
            >>> inn1 = tracker.get_innovation_number(1, 2, 'add_connection')
            >>> print(inn1)
            1
            >>> # Second genome also adds connection 1->2 in same generation
            >>> inn2 = tracker.get_innovation_number(1, 2, 'add_connection')
            >>> print(inn2)
            1
            >>> # Different connection gets different number
            >>> inn3 = tracker.get_innovation_number(1, 3, 'add_connection')
            >>> print(inn3)
            2
        """
        key = (input_node, output_node, mutation_type)
        
        # Check if this innovation already occurred this generation
        if key in self.generation_innovations:
            return self.generation_innovations[key]
        
        # New innovation - increment counter and record it
        self.global_counter += 1
        innovation_number = self.global_counter
        self.generation_innovations[key] = innovation_number
        
        return innovation_number
    
    def reset_generation(self):
        """
        Clear generation-specific tracking at the start of a new generation.
        
        This method should be called at the beginning of each generation's
        reproduction phase. It clears the generation_innovations dictionary
        but preserves the global_counter so innovation numbers never repeat.
        
        From the paper (p. 108):
        "By keeping a list of the innovations that occurred in the current
        generation..."
        
        Example:
            >>> tracker = InnovationTracker()
            >>> inn1 = tracker.get_innovation_number(1, 2)
            >>> print(inn1)
            1
            >>> tracker.reset_generation()  # Start new generation
            >>> # Same mutation in new generation gets NEW innovation number
            >>> inn2 = tracker.get_innovation_number(1, 2)
            >>> print(inn2)
            2
        """
        self.generation_innovations.clear()
    
    def get_current_innovation_number(self):
        """
        Get the current (most recently assigned) innovation number.
        
        Returns:
            int: The current value of the global counter
        """
        return self.global_counter
    
    def __repr__(self):
        return (f"InnovationTracker(global_counter={self.global_counter}, "
                f"generation_innovations={len(self.generation_innovations)} tracked)")
    
    def __getstate__(self):
        """
        Prepare tracker for pickling (checkpoint save).
        
        Returns a dictionary containing the state to be pickled.
        """
        return {
            'global_counter': self.global_counter,
            'generation_innovations': self.generation_innovations.copy()
        }
    
    def __setstate__(self, state):
        """
        Restore tracker from pickled state (checkpoint restore).
        
        Args:
            state: Dictionary containing the pickled state
        """
        self.global_counter = state['global_counter']
        self.generation_innovations = state['generation_innovations']
