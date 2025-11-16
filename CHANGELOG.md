# Changelog

All notable changes to neat-python will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added
- **Reproducibility Support**: Evolution can now be made deterministic by setting a random seed
  - Optional `seed` parameter in `[NEAT]` config section
  - Optional `seed` parameter in `Population.__init__()`
  - Optional `seed` parameter in `ParallelEvaluator.__init__()` for reproducible parallel evaluation
  - Per-genome deterministic seeding in parallel mode (seed + genome.key)
  - Comprehensive documentation in `docs/reproducibility.rst`
  - Complete test coverage in `tests/test_reproducibility.py` (9 tests)
  - **Fully backward compatible**: Existing code works without changes
  - Seed parameter controls Python's `random` module
  - Checkpoint system already preserved random state (unchanged)
  - All changes are fully backward compatible

- **Reproducibility Examples**: Demonstration scripts for reproducible NEAT evolution
  - Serial reproducibility example: `examples/xor/evolve-feedforward-reproducible.py`
    - Tests reproducibility (same seed → identical results)
    - Tests seed effect (different seeds → different evolution)
    - Tests backward compatibility (no seed parameter works)
  - Parallel reproducibility example: `examples/parallel-reproducible/`
    - `evolve-parallel.py` - Demonstrates parallel evaluation with reproducibility
    - `config-parallel` - Configuration file with seed parameter
    - `README.md` - Comprehensive documentation with best practices and troubleshooting
    - Tests parallel reproducibility (same seed + multiple workers → identical results)
    - Tests worker count independence (results consistent across worker counts)

- **New Example**: Inverted Double Pendulum example using Gymnasium
  - Complete example in `examples/inverted-double-pendulum/`
  - Uses `InvertedDoublePendulum-v5` environment (MuJoCo-based)
  - Demonstrates continuous control with 9-dimensional observation space
  - Includes evolution script with parallel evaluation
  - Includes test/visualization script for trained controllers
  - Full documentation in example README with usage tips

### Changed
- Dropped support for Python 3.6 and 3.7; neat-python now requires Python 3.8 or newer.
- Modernized internal implementation in `neat/` and `examples/` to use Python 3 features
  such as f-strings, comprehensions, dataclasses for internal helpers, and type hints,
  without changing the public API.

### Fixed
- **Population Size Drift**: Fixed small mismatches between actual population size and configured `pop_size`
  - `DefaultReproduction.reproduce()` now strictly enforces `len(population) == config.pop_size` for every non-extinction generation
  - New `_adjust_spawn_exact` helper adjusts per-species spawn counts after `compute_spawn()` to correct rounding/clamping drift
  - When adding individuals, extra genomes are allocated to smaller species first; when removing, genomes are taken from larger species first
  - Per-species minima (`min_species_size` and elitism) are always respected; invalid configurations (e.g., `pop_size < num_species * min_species_size`) raise a clear error
  - New tests in `tests/test_reproduction.py` ensure `DefaultReproduction.reproduce()` preserves exact population size over multiple generations
- **Orphaned Nodes Bug**: Fixed silent failure when nodes have no incoming connections after deletion mutations
  - `feed_forward_layers()` now correctly handles orphaned nodes (nodes with no incoming connections)
  - Orphaned nodes are treated as "bias neurons" that output `activation(bias)` independent of inputs
  - Placed in first evaluation layer as they are always ready
  - `required_for_output()` now includes orphaned nodes that feed into outputs
  - Aggregation functions (`max`, `min`, `maxabs`, `mean`, `median`) now handle empty inputs (return 0.0)
  - Comprehensive test coverage in `tests/test_graphs.py` (5 new tests) and `tests/test_nn.py` (integration test)

## [1.0.0] - 2025-01-09

### Added
- **Network Export**: JSON export capability for all network types (FeedForwardNetwork, RecurrentNetwork, CTRNN, IZNN)
  - New `neat.export` module with `export_network_json()` function
  - Framework-agnostic JSON format designed for conversion to ONNX, TensorFlow, PyTorch, etc.
  - Comprehensive format documentation in `docs/network-json-format.md`
  - Built-in function detection (activation/aggregation) vs. custom functions
  - Metadata support (fitness, generation, genome_id, custom fields)
  - Example demonstrating export workflow in `examples/export/export_example.py`
  - Full test suite in `tests/test_export.py`
  - No additional dependencies required (uses only Python standard library)

- **Innovation Number Tracking**: Full implementation of innovation numbers as described in the original NEAT paper (Stanley & Miikkulainen, 2002, Section 3.2)
  - Global innovation counter that increments across all generations
  - Same-generation deduplication of identical mutations
  - Innovation-based gene matching during crossover
  - Proper historical marking of genes for speciation
- New `InnovationTracker` class in `neat/innovation.py`
- Comprehensive unit tests in `tests/test_innovation.py` (19 tests)
- Integration tests in `tests/test_innovation_integration.py` (6 tests)
- Innovation tracking documentation in `INNOVATION_TRACKING_IMPLEMENTATION.md`

### Changed
- **BREAKING**: `DefaultConnectionGene.__init__()` now requires mandatory `innovation` parameter
- **BREAKING**: All connection gene creation must include innovation numbers
- **BREAKING**: Crossover now matches genes primarily by innovation number, not tuple keys
- **BREAKING**: Old checkpoints from pre-1.0 versions are incompatible with 1.0.0+
- `DefaultGenome.configure_crossover()` updated to match genes by innovation number per NEAT paper Figure 4
- All genome initialization methods assign innovation numbers to connections
- `DefaultReproduction` now creates and manages an `InnovationTracker` instance
- Checkpoint format updated to preserve innovation tracker state
- `ParallelEvaluator` now implements context manager protocol (`__enter__`/`__exit__`) for proper resource cleanup
- Improved resource management in `ParallelEvaluator` to prevent multiprocessing pool leaks
- Fixed `ParallelEvaluator.__del__()` to properly clean up resources without calling `terminate()` unnecessarily

### Removed
- **BREAKING**: `ThreadedEvaluator` has been removed
  - Reason: Minimal utility due to Python's Global Interpreter Lock (GIL)
  - Had implementation issues including unreliable cleanup and potential deadlocks
  - Migration: Use `ParallelEvaluator` for CPU-bound tasks
- **BREAKING**: `DistributedEvaluator` has been removed
  - Reason: Marked as beta/unstable with known reliability issues
  - Overly complex implementation (574 lines) with fragile error handling
  - Migration: Use established frameworks like Ray or Dask for distributed computing
- Removed `neat/threaded.py` module
- Removed `neat/distributed.py` module  
- Removed example files: `examples/xor/evolve-feedforward-threaded.py` and `examples/xor/evolve-feedforward-distributed.py`
- Removed test files: `tests/test_distributed.py` and `tests/test_xor_example_distributed.py`

### Added
- Context manager support for `ParallelEvaluator` - recommended usage pattern
- `ParallelEvaluator.close()` method for explicit resource cleanup
- New tests for `ParallelEvaluator` context manager functionality
- `MIGRATION.md` guide for users migrating from removed evaluators

### Migration
- See [MIGRATION.md](MIGRATION.md) for detailed guidance on updating existing code
- `ParallelEvaluator` remains fully backward compatible but context manager usage is recommended

## [0.93] - Previous Release

*Note: Changelog started with version 1.0. For changes prior to 1.0, please see git history.*

---

**For the complete migration guide, see [MIGRATION.md](MIGRATION.md)**
