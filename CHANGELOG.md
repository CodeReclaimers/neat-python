# Changelog

All notable changes to neat-python will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-01-09

### Added
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
