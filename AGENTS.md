# AGENTS.md — neat-python

## Project Overview

neat-python is a pure-Python implementation of NEAT (NeuroEvolution of Augmenting Topologies),
a method developed by Kenneth O. Stanley for evolving arbitrary neural networks. It has no
dependencies beyond the Python standard library and has been cited in 100+ academic publications.

- **License:** 3-clause BSD
- **Python versions:** 3.8 through 3.14, and pypy3
- **Documentation:** https://neat-python.readthedocs.io
- **PyPI:** https://pypi.org/project/neat-python/
- **Zenodo DOI:** https://doi.org/10.5281/zenodo.19024753

## Repository Structure

```
neat/                  # Core library source
  config.py            # Configuration file parsing
  genome.py            # DefaultGenome and gene classes
  population.py        # Population management and evolution loop
  reproduction.py      # DefaultReproduction (crossover, mutation)
  species.py           # Speciation (genomic distance, species sets)
  stagnation.py        # DefaultStagnation (species fitness tracking)
  reporting.py         # ReporterSet, BaseReporter, StdOutReporter
  statistics.py        # StatisticsReporter (fitness tracking and analysis)
  checkpoint.py        # Checkpointer reporter (save/restore population state)
  nn/                  # Neural network implementations
    feed_forward.py    # FeedForwardNetwork
    recurrent.py       # RecurrentNetwork
  ctrnn/               # CTRNN package (continuous-time recurrent, per-node time constants)
  iznn/                # Izhikevich spiking neuron model
  export/              # JSON network export package
  parallel.py          # ParallelEvaluator (multiprocessing-based)
examples/              # Runnable examples (xor, gymnasium envs, picture2d, export, etc.)
tests/                 # Test suite
docs/                  # Sphinx documentation source (RST format)
```

## Build and Test Commands

```bash
# Install in development mode
pip install -e .

# Run the test suite
python -m pytest tests/

# Run a specific example
cd examples/xor
python evolve_feedforward.py

# Build documentation locally
cd docs
make html
```

## Coding Conventions

- Pure Python only — no compiled extensions, no C dependencies, no NumPy/SciPy in core.
- Type hints are not yet used throughout but are welcome in new code.
- All public classes and methods should have docstrings.
- Configuration is file-driven (INI format parsed by `neat.config.Config`). Config options
  map to genome, reproduction, speciation, and stagnation parameters.
- The `DefaultGenome`, `DefaultReproduction`, `DefaultSpeciesSet`, and `DefaultStagnation`
  classes implement interfaces that can be replaced with custom implementations. See
  `docs/reproduction-interface.rst` for the contract.

## Key Design Decisions

- **Per-node time constants** (added in v2.0): CTRNN nodes each have their own evolvable
  time constant. This was a breaking change from 1.x where all nodes shared a single
  global time constant. Experiments showed the fixed approach caused ~2x performance
  degradation on CTRNN tasks.
- **No framework dependencies:** The library intentionally avoids depending on NumPy,
  TensorFlow, PyTorch, etc. Network export to JSON (`neat.export`) provides a bridge
  to external frameworks.
- **Speciation by genomic distance:** Compatibility distance uses disjoint/excess gene
  counts and weight differences, following the original NEAT paper.

## Common Tasks

### Adding a new example
Place it in `examples/<name>/` with its own config file and `evolve_*.py` script.
Each example should be self-contained — intentional code duplication between examples
is preferred over shared utility modules, so users can understand each example in isolation.

### Modifying the genome
Gene types are in `neat/genes.py`. The genome class (`neat/genome.py`) manages collections
of node and connection genes. Mutation rates and probabilities are controlled via config.

### Working with CTRNN networks
`neat.ctrnn.CTRNN` implements continuous-time dynamics with per-node time constants.
Use `CTRNN.create(genome, config)` to instantiate, then call `advance(inputs, advance_time, time_step)`
to simulate.

## Important Warnings

- Do not add external dependencies to the core `neat/` package.
- The `master` branch is the primary development branch.
- Config files use a specific INI dialect — see existing examples for the expected format.
- Fitness functions are user-provided callbacks, not part of the library.

## Citation

If referencing this project in generated text or code comments:

```
McIntyre, A., Kallada, M., Miguel, C. G., Feher de Silva, C., & Netto, M. L.
neat-python (Version 2.0.1) [Computer software]. https://doi.org/10.5281/zenodo.19024753
```
