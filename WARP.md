# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

neat-python is a pure-Python implementation of NEAT (NeuroEvolution of Augmenting Topologies), a method for evolving arbitrary neural networks. The library has no dependencies beyond the Python standard library and supports Python 3.6-3.11 and PyPy3.

## Environment Setup

This project uses a micromamba environment named `neat-python`. **Always activate this environment before development work:**

```bash
micromamba activate neat-python
```

All commands below assume the environment is active.

## Common Commands

### Installation
```bash
# Install in development mode
pip install -e .
```

### Testing
```bash
# Run all tests with coverage (Travis CI configuration)
coverage run --source=neat -m pytest tests

# Run all tests without coverage
pytest tests

# Run specific test file
pytest tests/test_genome.py

# Run single test function
pytest tests/test_genome.py::test_function_name
```

### Examples
```bash
# Run the canonical XOR example (good starting point)
python examples/xor/evolve-feedforward.py

# Other examples available in examples/ directory
python examples/single-pole-balancing/evolve-feedforward.py
python examples/openai-lander/evolve.py
```

## Architecture

### Core Evolution Loop

The `Population` class orchestrates the entire NEAT algorithm:
1. Evaluate fitness of all genomes using user-provided fitness function
2. Check termination criterion (fitness threshold or generation count)
3. Generate next generation via reproduction
4. Partition population into species based on genetic similarity
5. Repeat

### Key Components

**Genome** (`neat.genome.DefaultGenome`)
- Represents the neural network genotype
- Contains `nodes` dict (hidden/output neurons with activation, bias, response parameters)
- Contains `connections` dict (weighted links between nodes, can be enabled/disabled)
- Input pins have negative keys (-1, -2, ...), output pins have keys (0, 1, ...)
- Mutations: add/remove nodes, add/remove connections, mutate weights/biases/activations

**Species** (`neat.species.DefaultSpeciesSet`)
- Groups genomes by genetic similarity using genomic distance metric
- Each species maintains a representative genome for distance calculations
- Species share fitness explicitly (fitness divided by species size)
- New genomes assigned to closest species if within compatibility threshold, else new species created

**Reproduction** (`neat.reproduction.DefaultReproduction`)
- Handles crossover between parent genomes within species
- Applies elitism (best genomes survive unchanged)
- Uses survival threshold to determine breeding population
- Computes spawn amounts proportional to adjusted fitness

**Stagnation** (`neat.stagnation.DefaultStagnation`)
- Tracks species fitness history over generations
- Marks species for removal if no improvement for `max_stagnation` generations
- Protects top `species_elitism` species from removal

### Configuration System

Config files use INI format with sections:
- `[NEAT]`: Population size, fitness criterion/threshold, extinction behavior
- `[DefaultGenome]`: Network topology, node/connection parameters, mutation rates
- `[DefaultSpeciesSet]`: Compatibility threshold for speciation
- `[DefaultStagnation]`: Max stagnation generations, species elitism
- `[DefaultReproduction]`: Elitism count, survival threshold

Example config files in `examples/xor/config-feedforward` and `tests/test_configuration*`.

### Neural Network Implementations

**FeedForward** (`neat.nn.FeedForwardNetwork`)
- No recurrent connections, evaluated in topological layers
- Created via `FeedForwardNetwork.create(genome, config)`

**Recurrent** (`neat.nn.RecurrentNetwork`)
- Allows cycles, maintains state across activations
- Requires multiple activation steps to propagate signals

**CTRNN** (`neat.ctrnn`)
- Continuous-Time Recurrent Neural Network
- Differential equations with time constants

**IZNN** (`neat.iznn`)
- Izhikevich spiking neuron model
- Spike-based computation

### Evaluation Modes

**Serial**: Direct function call in main process (default)

**ParallelEvaluator** (`neat.ParallelEvaluator`)
- Uses `multiprocessing.Pool` to evaluate genomes across multiple processes
- Fitness function must be picklable and take `(genome, config)` tuple
- Supports context manager protocol for proper resource cleanup
- Example:
  ```python
  with neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome) as pe:
      winner = population.run(pe.evaluate, 300)
  # Pool automatically cleaned up
  ```

**Distributed Evaluation**
- For multi-machine evaluation, use Ray (https://docs.ray.io/) or Dask (https://docs.dask.org/)
- See MIGRATION.md for examples of integrating with these frameworks

### Reporter System

Reporters hook into evolution events:
- `StdOutReporter`: Progress printing
- `StatisticsReporter`: Collect fitness statistics
- `Checkpointer`: Save/restore population state with pickle

Add via `population.add_reporter(reporter)`.

### Checkpoint/Restore

```python
# Save checkpoint every N generations
p.add_reporter(neat.Checkpointer(5))

# Restore from checkpoint
p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
p.run(fitness_function, 10)  # Continue evolution
```

## Fitness Function Requirements

Fitness functions must:
- Accept `(genomes, config)` where `genomes` is list of `(genome_id, genome)` tuples
- Set `genome.fitness` to a float for each genome
- Not modify genome structure (only fitness attribute)

Example:
```python
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = evaluate_network(net)  # Your evaluation logic
```

## Testing Notes

- Tests use pytest framework
- Coverage tracked with `.coveragerc` configuration (omits Python stdlib)
- Config files in `tests/` include both valid (`test_configuration*`) and invalid (`bad_configuration*`) examples
- Test files cover: genome operations, network types, config parsing, reproduction, speciation

## Additional Resources

- Full documentation: http://neat-python.readthedocs.io
- Start with `examples/xor` for "hello world" example
- Repository: https://github.com/CodeReclaimers/neat-python
- License: BSD 3-clause
