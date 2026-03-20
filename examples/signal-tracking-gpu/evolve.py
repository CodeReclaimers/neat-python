"""
CTRNN Signal Tracking with CPU vs GPU Performance Comparison

Evolves a Continuous-Time Recurrent Neural Network (CTRNN) to perform frequency
doubling: given sin(2*pi*t) and cos(2*pi*t) as inputs, produce sin(4*pi*t) as
output. This task requires nonlinear transformation of the inputs (since
sin(2x) = 2*sin(x)*cos(x)) and is a natural fit for CTRNN dynamics.

The example runs evolution using both the pure-Python CPU evaluator and the
GPU-accelerated evaluator (when available), printing a timing comparison.

Usage:
    python evolve.py                      # CPU only (GPU if available)
    python evolve.py --cpu-only           # Force CPU only
    python evolve.py --gpu-only           # Force GPU only (requires CuPy)
    python evolve.py --generations 100    # Set number of generations
    python evolve.py --pop-size 500       # Override population size
"""

import argparse
import math
import os
import time

import neat

# ---------------------------------------------------------------------------
# Task definition: frequency doubling
#
#   inputs:  [sin(2*pi*t), cos(2*pi*t)]
#   target:  sin(4*pi*t) = 2 * sin(2*pi*t) * cos(2*pi*t)
#
# The integration runs for t_max seconds at time step dt. Both CPU and GPU
# evaluators use identical input signals and fitness computation for a fair
# comparison.
# ---------------------------------------------------------------------------

DT = 0.01           # Integration time step (seconds)
T_MAX = 1.0         # Total simulation time (seconds)
NUM_STEPS = int(T_MAX / DT)   # 100 steps
FREQ = 1.0          # Base frequency (Hz)

# Precompute the target trajectory for fitness evaluation.
TARGET = [math.sin(4.0 * math.pi * FREQ * (step * DT)) for step in range(NUM_STEPS)]


def input_fn(t, dt):
    """Return the two input signals at time t."""
    return [math.sin(2.0 * math.pi * FREQ * t),
            math.cos(2.0 * math.pi * FREQ * t)]


# ---------------------------------------------------------------------------
# CPU evaluation
# ---------------------------------------------------------------------------

def eval_genomes_cpu(genomes, config):
    """Evaluate all genomes sequentially on CPU using neat.ctrnn.CTRNN."""
    for genome_id, genome in genomes:
        net = neat.ctrnn.CTRNN.create(genome, config)
        net.reset()

        total_se = 0.0
        unstable = False
        for step in range(NUM_STEPS):
            t = step * DT
            inputs = input_fn(t, DT)
            output = net.advance(inputs, DT, DT)

            if math.isnan(output[0]) or math.isinf(output[0]) or abs(output[0]) > 1e10:
                unstable = True
                break

            total_se += (output[0] - TARGET[step]) ** 2

        if unstable:
            genome.fitness = -10.0
        else:
            genome.fitness = -total_se / NUM_STEPS


# ---------------------------------------------------------------------------
# GPU evaluation
# ---------------------------------------------------------------------------

def make_gpu_evaluator():
    """Create a GPUCTRNNEvaluator with the same task parameters."""
    import numpy as np
    from neat.gpu.evaluator import GPUCTRNNEvaluator

    target_np = np.array(TARGET, dtype=np.float32)

    def fitness_fn(trajectory):
        """Negative mean squared error over the output trajectory.

        trajectory: ndarray of shape [num_steps, num_outputs].
        """
        output = trajectory[:, 0]
        mse = float(np.mean((output - target_np) ** 2))
        return -mse

    return GPUCTRNNEvaluator(
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

    # Collect per-generation timing via a custom reporter.
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
        description='CTRNN signal tracking with CPU vs GPU comparison')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Run CPU evaluation only')
    parser.add_argument('--gpu-only', action='store_true',
                        help='Run GPU evaluation only (requires CuPy)')
    parser.add_argument('--generations', type=int, default=50,
                        help='Number of generations (default: 50)')
    parser.add_argument('--pop-size', type=int, default=None,
                        help='Override population size from config')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ctrnn')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
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
    print('CTRNN Signal Tracking — Frequency Doubling')
    print('=' * 65)
    print(f'Task:        sin(2*pi*t), cos(2*pi*t) -> sin(4*pi*t)')
    print(f'Simulation:  dt={DT}s, t_max={T_MAX}s, {NUM_STEPS} steps')
    print(f'Population:  {config.pop_size}')
    print(f'Generations: {args.generations}')
    print(f'Seed:        {args.seed}')
    print(f'GPU:         {"available" if gpu_ok else "not available"}')
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
              f'best fitness = {cpu_winner.fitness:.6f}')

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
              f'best fitness = {gpu_winner.fitness:.6f}')

    # --- Comparison ---
    if cpu_result and gpu_result:
        cpu_winner, cpu_times, cpu_total = cpu_result
        gpu_winner, gpu_times, gpu_total = gpu_result

        # Compute evaluation-only time (subtract a rough estimate of NEAT
        # overhead by noting that reproduction/speciation is identical).
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
        print(f'{"Best fitness":>20} {cpu_winner.fitness:>12.6f} {gpu_winner.fitness:>12.6f}')
        print()
        print(f'Note: GPU speedup increases with larger populations. '
              f'Try --pop-size 1000.')


if __name__ == '__main__':
    main()
