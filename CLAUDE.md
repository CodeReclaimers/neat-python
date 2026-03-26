# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NEAT-Python is a pure-Python implementation of NEAT (NeuroEvolution of Augmenting Topologies), a method for evolving arbitrary neural networks. This project has no dependencies beyond the Python standard library and supports Python 3.8-3.14 and PyPy3.

## Development Environment

A micromamba environment named `neat-python` is available with all dependencies (including CuPy for GPU tests). Prefix commands with `micromamba run -n neat-python` to use it:

```bash
# Run commands in the neat-python environment
micromamba run -n neat-python pytest tests/ -v
micromamba run -n neat-python python examples/xor/evolve-minimal.py
```

## Key Development Commands

All commands below assume the `neat-python` micromamba environment. Prefix with `micromamba run -n neat-python` when running from outside the environment.

### Testing
```bash
# Run all tests with coverage
micromamba run -n neat-python pytest tests/ --cov=neat --cov-report=term --cov-report=xml -v

# Run a single test file
micromamba run -n neat-python pytest tests/test_genome.py -v

# Run a specific test
micromamba run -n neat-python pytest tests/test_genome.py::TestGenome::test_mutate_add_node -v

# Run GPU tests (requires CuPy + NVIDIA GPU, both available in the micromamba env)
micromamba run -n neat-python pytest tests/test_gpu.py -v
```

### Installation
```bash
# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[dev]"        # Development tools (pytest, coverage, etc.)
pip install -e ".[docs]"       # Documentation building tools
pip install -e ".[examples]"   # Dependencies for running examples
pip install -e ".[all]"        # Everything

# Build distributions (requires build package)
python -m build
```

### Documentation
```bash
# Build documentation (from docs/ directory)
cd docs
make clean && make html

# View built docs at docs/_build/html/index.html
```

### Running Examples
```bash
# Start with the simplest example
python examples/xor/evolve-minimal.py

# Other examples demonstrate various NEAT features
python examples/xor/evolve-feedforward.py
python examples/xor/evolve-feedforward-parallel.py
python examples/single-pole-balancing/evolve-feedforward.py
```

## Architecture Overview

### Core Evolution Loop (neat/population.py)
The `Population` class implements the main NEAT algorithm:
1. Evaluate fitness of all genomes
2. Check termination criterion
3. Generate next generation via reproduction
4. Partition into species based on genetic similarity
5. Repeat

The population accepts a fitness function that takes `(genomes, config)` as arguments and assigns a fitness value to each genome.

### Genome Representation (neat/genome.py)
- **DefaultGenome**: Represents a single individual/network
- Consists of node genes (neurons) and connection genes (synapses)
- Supports various mutation operators: add/delete nodes, add/delete connections, mutate weights
- Configuration controls network topology (feed-forward vs recurrent, initial connectivity)

### Innovation Tracking (neat/innovation.py)
- **InnovationTracker**: Ensures consistent numbering of genes across population
- Critical for proper crossover - matching genes have same innovation numbers
- Tracks both node and connection innovations separately
- Reset at each generation to maintain consistency

### Speciation (neat/species.py)
- **DefaultSpeciesSet**: Groups similar genomes into species
- Uses genetic distance metric based on disjoint/excess genes and weight differences
- Each species has a representative that determines membership
- Protects innovation by allowing new structures time to optimize

### Reproduction (neat/reproduction.py)
- **DefaultReproduction**: Handles creation of offspring
- Implements crossover (combining parent genomes) and mutation
- Allocates offspring to species based on fitness
- Manages stagnation removal of underperforming species

### Network Implementations (neat/nn/)
- **FeedForwardNetwork**: Fast implementation for acyclic networks
- **RecurrentNetwork**: Supports networks with cycles
- Both created from evolved genomes via `.create(genome, config)`

### Specialized Neuron Models
- **neat/ctrnn/**: Continuous-Time Recurrent Neural Networks
- **neat/iznn/**: Izhikevich spiking neuron model

### Configuration System (neat/config.py)
- Config files are INI format with sections like `[DefaultGenome]`, `[DefaultReproduction]`
- The `Config` class loads and validates all parameters
- Each major component has its own config class (e.g., `DefaultGenomeConfig`)
- Config parameter types are strictly enforced

### Parallel Evaluation (neat/parallel.py)
- **ParallelEvaluator**: Uses multiprocessing to evaluate fitness in parallel
- Pass it to population.run() instead of a regular fitness function
- Example: `pe = ParallelEvaluator(num_workers, eval_genome); p.run(pe.evaluate)`

### Network Export (neat/export/)
- Exports trained networks to JSON format for interoperability
- Framework-agnostic format with metadata support
- See `docs/network-json-format.md` for format specification
- Example conversion tools in `examples/export/`

## Configuration Files

All examples include a config file (e.g., `config-feedforward`) that specifies:
- Network structure (num_inputs, num_outputs, num_hidden)
- Mutation rates (node_add_prob, conn_add_prob, etc.)
- Activation functions, aggregation functions
- Speciation parameters (compatibility threshold)
- Reproduction parameters (elitism, survival threshold)
- Population size and fitness criterion

## Common Patterns

### Basic NEAT Run
```python
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-file')
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(False))
winner = p.run(eval_genomes, n=300)  # Run for max 300 generations
```

### Reproducibility
Set random seed for deterministic evolution:
```python
p = neat.Population(config, seed=42)
# or in config file: seed = 42
```

### Checkpointing
```python
p.add_reporter(neat.Checkpointer(generation_interval=10))
# Restore from checkpoint
p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-50')
```

### Custom Reporters
Reporters receive callbacks for events like generation start/end, species creation, etc. See `neat/reporting.py` for the interface.

## Progress Logs

Do NOT commit `progress-*.md` files to this repository. They are local working notes excluded via `.gitignore`. Continue writing them locally per the global instructions, but do not `git add` them.

## Testing Notes

- Test files use configuration files in `tests/` directory (e.g., `test_configuration`)
- Many tests verify evolutionary properties statistically (may have rare failures)
- Coverage is tracked via coveralls
- Tests include edge cases for graph cycles, gene mutations, and crossover operations

## Important Implementation Details

### Feed-forward Networks
- `initial_connection` config determines starting topology
- `feed_forward = True` enables cycle detection to enforce acyclic structure
- Feed-forward networks are evaluated in a single pass

### Recurrent Networks
- `feed_forward = False` allows cycles
- Require multiple activation steps to propagate signals
- Use time_constant parameter for CTRNN variant

### Structural Mutations
- Adding a node splits an existing connection
- `single_structural_mutation = True` limits to one structural change per mutation
- Disabled connections can be re-enabled by mutation

### Crossover
- Only occurs between genomes in same species
- Matching genes (same innovation number) are randomly selected from either parent
- Disjoint/excess genes come from more fit parent
- See `test_genome_crossover.py` for detailed test cases
