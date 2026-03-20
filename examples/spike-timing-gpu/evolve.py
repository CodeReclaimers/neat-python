"""
Izhikevich Spike Pattern Discrimination with CPU vs GPU Performance Comparison

Evolves a spiking neural network using the Izhikevich neuron model to perform
temporal pattern discrimination. Two input neurons receive alternating current
pulses in four phases (neither, input 0 only, input 1 only, both). The network
must learn to spike output 0 when only input 0 is active, and output 1 when
only input 1 is active — essentially an XOR-like temporal discrimination task
where the network must distinguish which input is driving it.

The example runs evolution using both the pure-Python CPU evaluator and the
GPU-accelerated evaluator (when available), printing a timing comparison.

Usage:
    python evolve.py                      # CPU + GPU (if available)
    python evolve.py --cpu-only           # Force CPU only
    python evolve.py --gpu-only           # Force GPU only (requires CuPy)
    python evolve.py --generations 200    # Set number of generations
    python evolve.py --pop-size 500       # Override population size
"""

import argparse
import os
import time

import neat

# ---------------------------------------------------------------------------
# Task definition: temporal spike pattern discrimination
#
# The simulation runs for t_max milliseconds. The input signal is divided into
# four equal phases:
#
#   Phase 0 (0 - 12.5 ms):    both inputs off    -> neither output should spike
#   Phase 1 (12.5 - 25 ms):   input 0 on         -> output 0 should spike
#   Phase 2 (25 - 37.5 ms):   input 1 on         -> output 1 should spike
#   Phase 3 (37.5 - 50 ms):   both inputs on      -> neither output should spike
#
# This is essentially a temporal XOR: each output should respond selectively
# to its corresponding input, but NOT when both inputs are active.
#
# Fitness rewards correct spikes and penalizes incorrect ones.
# ---------------------------------------------------------------------------

DT = 0.05            # Integration time step (milliseconds)
T_MAX = 50.0         # Total simulation time (milliseconds)
NUM_STEPS = int(T_MAX / DT)   # 1000 steps

# Input current amplitude (sufficient to drive regular spiking neurons).
INPUT_CURRENT = 15.0

# Phase boundaries (in milliseconds).
PHASE_DURATION = T_MAX / 4.0
PHASES = [
    (0.0,                  PHASE_DURATION),        # Phase 0: both off
    (PHASE_DURATION,       2 * PHASE_DURATION),     # Phase 1: input 0 on
    (2 * PHASE_DURATION,   3 * PHASE_DURATION),     # Phase 2: input 1 on
    (3 * PHASE_DURATION,   4 * PHASE_DURATION),     # Phase 3: both on
]


def input_fn(t, dt):
    """Return input currents [i0, i1] at time t (milliseconds)."""
    if t < PHASE_DURATION:
        return [0.0, 0.0]                     # Phase 0: both off
    elif t < 2 * PHASE_DURATION:
        return [INPUT_CURRENT, 0.0]           # Phase 1: input 0 only
    elif t < 3 * PHASE_DURATION:
        return [0.0, INPUT_CURRENT]           # Phase 2: input 1 only
    else:
        return [INPUT_CURRENT, INPUT_CURRENT]  # Phase 3: both on


def compute_fitness_from_spikes(spike_counts):
    """Compute fitness from per-phase spike counts.

    spike_counts: dict mapping (phase, output) -> int

    Scoring:
      - Phase 0: reward no spikes from either output       (+1 each, max +2)
      - Phase 1: reward spikes from output 0, none from 1  (+1 each, max +2)
      - Phase 2: reward spikes from output 1, none from 0  (+1 each, max +2)
      - Phase 3: reward no spikes from either output       (+1 each, max +2)

    Maximum fitness = 8.0. A "spike" means count > 0 for reward, count == 0
    for silence reward.
    """
    fitness = 0.0

    # Phase 0: silence from both.
    if spike_counts[(0, 0)] == 0:
        fitness += 1.0
    if spike_counts[(0, 1)] == 0:
        fitness += 1.0

    # Phase 1: output 0 fires, output 1 silent.
    if spike_counts[(1, 0)] > 0:
        fitness += 1.0
    if spike_counts[(1, 1)] == 0:
        fitness += 1.0

    # Phase 2: output 1 fires, output 0 silent.
    if spike_counts[(2, 0)] == 0:
        fitness += 1.0
    if spike_counts[(2, 1)] > 0:
        fitness += 1.0

    # Phase 3: silence from both.
    if spike_counts[(3, 0)] == 0:
        fitness += 1.0
    if spike_counts[(3, 1)] == 0:
        fitness += 1.0

    return fitness


def get_phase(t):
    """Return the phase index (0-3) for time t in milliseconds."""
    for i, (start, end) in enumerate(PHASES):
        if start <= t < end:
            return i
    return 3  # Last phase for t == T_MAX


# ---------------------------------------------------------------------------
# CPU evaluation
# ---------------------------------------------------------------------------

def eval_genomes_cpu(genomes, config):
    """Evaluate all genomes sequentially on CPU using neat.iznn.IZNN."""
    for genome_id, genome in genomes:
        net = neat.iznn.IZNN.create(genome, config)
        net.reset()
        net.set_inputs(input_fn(0.0, DT))

        spike_counts = {}
        for phase in range(4):
            for out in range(2):
                spike_counts[(phase, out)] = 0

        for step in range(NUM_STEPS):
            t = step * DT
            net.set_inputs(input_fn(t, DT))
            output = net.advance(DT)

            phase = get_phase(t)
            for out_idx in range(2):
                if output[out_idx] > 0:
                    spike_counts[(phase, out_idx)] += 1

        genome.fitness = compute_fitness_from_spikes(spike_counts)


# ---------------------------------------------------------------------------
# GPU evaluation
# ---------------------------------------------------------------------------

def make_gpu_evaluator():
    """Create a GPUIZNNEvaluator with the same task parameters."""
    import numpy as np
    from neat.gpu.evaluator import GPUIZNNEvaluator

    # Precompute phase boundaries in step indices.
    phase_bounds = []
    for start, end in PHASES:
        s_step = int(start / DT)
        e_step = int(end / DT)
        phase_bounds.append((s_step, e_step))

    def fitness_fn(trajectory):
        """Compute fitness from the spike trajectory.

        trajectory: ndarray of shape [num_steps, num_outputs] with 0/1 values.
        """
        spike_counts = {}
        for phase in range(4):
            for out in range(2):
                s, e = phase_bounds[phase]
                spike_counts[(phase, out)] = int(np.sum(trajectory[s:e, out]))

        return compute_fitness_from_spikes(spike_counts)

    return GPUIZNNEvaluator(
        dt=DT,
        t_max=T_MAX,
        input_fn=input_fn,
        fitness_fn=fitness_fn,
    )


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------

def run_evolution(config, eval_fn, n_generations, label, seed=42):
    """Run NEAT evolution and return (winner, per-generation times, total time)."""
    pop = neat.Population(config, seed=seed)

    gen_times = []

    class TimingReporter(neat.reporting.BaseReporter):
        def __init__(self):
            self._gen_start = None

        def start_generation(self, generation):
            self._gen_start = time.perf_counter()

        def post_evaluate(self, config, population, species, best_genome):
            elapsed = time.perf_counter() - self._gen_start
            gen_times.append(elapsed)

    pop.add_reporter(TimingReporter())
    pop.add_reporter(neat.StdOutReporter(False))

    t0 = time.perf_counter()
    winner = pop.run(eval_fn, n_generations)
    total = time.perf_counter() - t0

    return winner, gen_times, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Izhikevich spike pattern discrimination with CPU vs GPU comparison')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Run CPU evaluation only')
    parser.add_argument('--gpu-only', action='store_true',
                        help='Run GPU evaluation only (requires CuPy)')
    parser.add_argument('--generations', type=int, default=100,
                        help='Number of generations (default: 100)')
    parser.add_argument('--pop-size', type=int, default=None,
                        help='Override population size from config')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-iznn')
    config = neat.Config(neat.iznn.IZGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    if args.pop_size is not None:
        config.pop_size = args.pop_size

    # Check GPU availability.
    gpu_ok = False
    if not args.cpu_only:
        try:
            from neat.gpu import gpu_available
            gpu_ok = gpu_available()
        except ImportError:
            pass

    if args.gpu_only and not gpu_ok:
        print("ERROR: --gpu-only requested but CuPy/CUDA is not available.")
        print("Install with: pip install 'neat-python[gpu]'")
        return

    # Banner.
    print('=' * 65)
    print('Izhikevich Spike Pattern Discrimination')
    print('=' * 65)
    print(f'Task:        Temporal XOR — selective spiking per input phase')
    print(f'Phases:      off/off, on/off, off/on, on/on '
          f'({PHASE_DURATION:.1f} ms each)')
    print(f'Simulation:  dt={DT} ms, t_max={T_MAX} ms, {NUM_STEPS} steps')
    print(f'Population:  {config.pop_size}')
    print(f'Generations: {args.generations}')
    print(f'Seed:        {args.seed}')
    print(f'GPU:         {"available" if gpu_ok else "not available"}')
    print(f'Max fitness: 8.0 (1 point per correct output per phase)')
    print()

    cpu_result = None
    gpu_result = None

    # --- CPU run ---
    if not args.gpu_only:
        print('-' * 65)
        print('Running CPU evaluation...')
        print('-' * 65)
        cpu_winner, cpu_times, cpu_total = run_evolution(
            config, eval_genomes_cpu, args.generations, 'CPU', seed=args.seed)
        cpu_result = (cpu_winner, cpu_times, cpu_total)
        print(f'\nCPU: {cpu_total:.2f}s total, '
              f'{sum(cpu_times)/len(cpu_times):.4f}s/gen avg, '
              f'best fitness = {cpu_winner.fitness:.1f}/8.0')

    # --- GPU run ---
    if gpu_ok and not args.cpu_only:
        print()
        print('-' * 65)
        print('Running GPU evaluation...')
        print('-' * 65)
        gpu_eval = make_gpu_evaluator()
        gpu_winner, gpu_times, gpu_total = run_evolution(
            config, gpu_eval.evaluate, args.generations, 'GPU', seed=args.seed)
        gpu_result = (gpu_winner, gpu_times, gpu_total)
        print(f'\nGPU: {gpu_total:.2f}s total, '
              f'{sum(gpu_times)/len(gpu_times):.4f}s/gen avg, '
              f'best fitness = {gpu_winner.fitness:.1f}/8.0')

    # --- Comparison ---
    if cpu_result and gpu_result:
        cpu_winner, cpu_times, cpu_total = cpu_result
        gpu_winner, gpu_times, gpu_total = gpu_result

        cpu_eval_avg = sum(cpu_times) / len(cpu_times)
        gpu_eval_avg = sum(gpu_times) / len(gpu_times)
        speedup = cpu_total / gpu_total if gpu_total > 0 else float('inf')
        eval_speedup = cpu_eval_avg / gpu_eval_avg if gpu_eval_avg > 0 else float('inf')

        print()
        print('=' * 65)
        print('Performance Comparison')
        print('=' * 65)
        print(f'{"":>20} {"CPU":>12} {"GPU":>12} {"Speedup":>12}')
        print(f'{"":>20} {"---":>12} {"---":>12} {"-------":>12}')
        print(f'{"Total time":>20} {cpu_total:>11.2f}s {gpu_total:>11.2f}s '
              f'{speedup:>10.1f}x')
        print(f'{"Avg per generation":>20} {cpu_eval_avg:>11.4f}s {gpu_eval_avg:>11.4f}s '
              f'{eval_speedup:>10.1f}x')
        print(f'{"Best fitness":>20} {cpu_winner.fitness:>10.1f}/8 '
              f'{gpu_winner.fitness:>10.1f}/8')
        print()
        print(f'Note: The Izhikevich model runs {NUM_STEPS} integration steps '
              f'per genome, making')
        print(f'      GPU batching especially effective. '
              f'Try --pop-size 1000 for larger speedups.')


if __name__ == '__main__':
    main()
