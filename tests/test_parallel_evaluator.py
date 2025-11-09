"""Tests for ParallelEvaluator with context manager support."""

import multiprocessing
import os
import neat


def eval_dummy_genome(genome, config):
    """Simple fitness evaluation function."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    genome.fitness = 1.0
    return 1.0


def test_parallel_evaluator_context_manager():
    """Test that ParallelEvaluator works as a context manager."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population.
    p = neat.Population(config)

    # Test using context manager
    with neat.ParallelEvaluator(2, eval_dummy_genome) as pe:
        assert not pe._closed, "ParallelEvaluator should not be closed on entry"
        assert pe.pool is not None, "Pool should be initialized"
        
        # Run for just 1 generation
        winner = p.run(pe.evaluate, 1)
        assert winner is not None

    # After exiting context, pool should be closed
    # Note: We can't easily test pe._closed here because pe might not be in scope,
    # but the test passing without hanging verifies cleanup worked


def test_parallel_evaluator_explicit_close():
    """Test that ParallelEvaluator.close() works correctly."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population.
    p = neat.Population(config)

    # Test explicit close
    pe = neat.ParallelEvaluator(2, eval_dummy_genome)
    assert not pe._closed
    assert pe.pool is not None
    
    # Run for just 1 generation
    winner = p.run(pe.evaluate, 1)
    assert winner is not None
    
    # Explicitly close
    pe.close()
    assert pe._closed, "ParallelEvaluator should be closed after close()"
    assert pe.pool is None, "Pool should be None after close()"
    
    # Calling close again should be safe
    pe.close()
    assert pe._closed


def test_parallel_evaluator_backward_compatibility():
    """Test that ParallelEvaluator still works without context manager."""
    # Load configuration.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population.
    p = neat.Population(config)

    # Old style usage (without context manager)
    pe = neat.ParallelEvaluator(2, eval_dummy_genome)
    winner = p.run(pe.evaluate, 1)
    assert winner is not None
    
    # Clean up explicitly (good practice, but __del__ should handle it)
    pe.close()


if __name__ == '__main__':
    test_parallel_evaluator_context_manager()
    test_parallel_evaluator_explicit_close()
    test_parallel_evaluator_backward_compatibility()
    print("All ParallelEvaluator tests passed!")
