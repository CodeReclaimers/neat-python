# Parallel Evaluation with Reproducibility

This example demonstrates how to use parallel evaluation with NEAT while maintaining reproducibility through deterministic seeding.

## Overview

Parallel evaluation can significantly speed up fitness computation when evaluating many genomes. However, using multiple processes introduces a challenge: how do we ensure reproducible results?

The answer is **per-genome deterministic seeding**. When you specify a seed in the `ParallelEvaluator`, each genome gets a unique but deterministic seed derived from:

```
genome_seed = base_seed + genome.key
```

This ensures:
- **Reproducibility**: Same seed produces identical evolution across runs
- **Determinism**: Same genome always gets the same fitness (given same random seed)
- **Independence**: Different genomes get different random sequences
- **Consistency**: Results are identical regardless of worker count

## Key Features

- **Parallel Evaluation**: Uses `ParallelEvaluator` to evaluate genomes across multiple CPU cores
- **Reproducibility**: Setting a seed ensures deterministic results
- **Verification**: Script demonstrates reproducibility with multiple tests
- **Cross-platform**: Works consistently across different systems and worker counts

## Configuration

The `config-parallel` file includes standard NEAT parameters. To enable reproducibility, uncomment the seed parameter:

```ini
[NEAT]
fitness_criterion     = max
fitness_threshold     = 100.0
pop_size              = 150
reset_on_extinction   = False
no_fitness_termination = False

# Uncomment to enable reproducibility:
seed = 42
```

Alternatively, you can set the seed programmatically:

```python
pop = neat.Population(config, seed=42)
with neat.ParallelEvaluator(4, eval_genome, seed=42) as pe:
    winner = pop.run(pe.evaluate, 100)
```

## Running the Example

```bash
python evolve-parallel.py
```

The script will automatically detect the number of CPU cores and perform three tests:

1. **Parallel Reproducibility Test**: Verifies that the same seed with multiple workers produces identical results across runs
2. **Different Seeds Test**: Shows that different seeds lead to different evolution paths
3. **Worker Count Independence Test**: Confirms results are consistent with different numbers of workers

## Expected Output

When you run the example, you'll see:

```
======================================================================
  TEST 1: PARALLEL REPRODUCIBILITY (4 workers)
======================================================================

Running first parallel evolution with seed=42...
Running second parallel evolution with seed=42...

======================================================================
  RESULTS
======================================================================

Run 1 - Fitness: 234.50, Generations: 25, Nodes: 12, Connections: 18
Run 2 - Fitness: 234.50, Generations: 25, Nodes: 12, Connections: 18

✓ Fitness match:     True
✓ Generation match:  True
✓ Structure match:   True

✅ SUCCESS: Reproducibility with parallel evaluation works!
```

## How It Works

### Seed Propagation

1. You specify a base seed when creating `ParallelEvaluator`:
   ```python
   with neat.ParallelEvaluator(num_workers, eval_genome, seed=42) as pe:
   ```

2. Each genome gets a deterministic seed: `seed + genome.key`

3. The wrapper function (`_eval_wrapper` in `neat/parallel.py`) sets the seed before evaluating the genome

### Reproducible Parallel Evaluation

```python
import neat

def eval_genome(genome, config):
    # This uses Python's random module
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # ... fitness evaluation ...
    return fitness

# Create population and evaluator with same seed
config = neat.Config(...)
pop = neat.Population(config, seed=42)

with neat.ParallelEvaluator(4, eval_genome, seed=42) as pe:
    winner = pop.run(pe.evaluate, 100)
```

Both the Population and ParallelEvaluator use seed=42, ensuring:
- Population initialization is deterministic
- Mutation operations use seeded randomness
- Each genome's fitness evaluation uses deterministic randomness

## Limitations

### Python's Random Module Only

The seed parameter controls only Python's `random` module. If your fitness function uses other RNGs:

```python
import numpy as np
import torch

def eval_genome(genome, config):
    # These need separate seeding:
    random.seed(42)           # Already handled by ParallelEvaluator
    np.random.seed(42)        # Need to do this manually
    torch.manual_seed(42)     # Need to do this manually
    
    # ... fitness evaluation ...
    return fitness
```

### Non-Deterministic Fitness Functions

Some fitness evaluations are inherently non-deterministic:
- External simulators with their own RNG
- Network operations (timing-dependent)
- Hardware behavior
- File I/O operations

In these cases, perfect reproducibility may not be achievable.

### Worker Pool Initialization

Each worker process inherits the random state from the parent process at fork time. The wrapper function reseeds before evaluation, but:
- Global state in worker processes won't be reset
- Any initialization code runs once per worker
- This is generally not a problem (just be aware)

## Best Practices

### Development and Debugging

Use a fixed seed during development:

```python
# Fixed seed for reproducibility
pop = neat.Population(config, seed=42)
with neat.ParallelEvaluator(4, eval_genome, seed=42) as pe:
    winner = pop.run(pe.evaluate, 100)
```

### Scientific Evaluation

For final results, run with multiple seeds:

```python
import statistics

results = []
for seed in [42, 123, 456, 789, 999]:
    pop = neat.Population(config, seed=seed)
    with neat.ParallelEvaluator(4, eval_genome, seed=seed) as pe:
        winner = pop.run(pe.evaluate, 100)
    results.append(winner.fitness)

print(f"Mean fitness: {statistics.mean(results):.2f}")
print(f"Std dev:      {statistics.stdev(results):.2f}")
```

### Performance Testing

Verify reproducibility before and after optimizations:

```python
# Baseline
baseline_result = run_with_seed(42, 4)

# After optimization
optimized_result = run_with_seed(42, 4)

assert baseline_result == optimized_result, "Optimization broke reproducibility!"
```

## Troubleshooting

### Results Don't Match

If you get different results with the same seed:

1. **Check seed parameter**: Verify both Population and ParallelEvaluator use the same seed
2. **Check fitness function**: Ensure your fitness function is deterministic given the same seed
3. **Check other RNGs**: If using NumPy, PyTorch, etc., seed those too
4. **Platform differences**: Different OS/Python versions may produce different random sequences

### Performance Issues

If parallel evaluation is slower than serial:

1. **Check fitness computation time**: Parallel evaluation has overhead; only benefits if fitness evaluation is significant
2. **Check worker count**: Too many workers = too much overhead; too few = underutilization
3. **Use profiling**: `cProfile` can help identify bottlenecks

## Files

- **evolve-parallel.py**: Main script with reproducibility tests
- **config-parallel**: NEAT configuration file
- **README.md**: This file

## References

- [NEAT-Python Documentation](http://neat-python.readthedocs.io/)
- [Reproducibility Guide](../../docs/reproducibility.rst)
- [ParallelEvaluator API](https://neat-python.readthedocs.io/en/latest/module_summaries.html#neat.parallel.ParallelEvaluator)

## Notes

- The fitness function in this example is simplified for demonstration
- In real applications, replace with your actual neural network evaluation
- Reproducibility requires setting seed for all RNG sources used
- Worker count independence is guaranteed by per-genome seeding strategy
