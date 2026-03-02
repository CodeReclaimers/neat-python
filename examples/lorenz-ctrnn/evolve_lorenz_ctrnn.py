"""
Lorenz Attractor CTRNN Prediction

Evolves a Continuous-Time Recurrent Neural Network (CTRNN) using NEAT to predict
the next-step state of the Lorenz attractor. The Lorenz system is a chaotic
dynamical system where small errors grow exponentially, making it a challenging
benchmark for evolved networks.

Ported from the Julia NeatEvolution lorenz_ctrnn example.

Usage:
    python evolve_lorenz_ctrnn.py                          # base mode, 3 outputs
    python evolve_lorenz_ctrnn.py --mode products          # 6 inputs with xy, xz, yz
    python evolve_lorenz_ctrnn.py --mode product-agg       # product aggregation
    python evolve_lorenz_ctrnn.py --z-only                 # predict only z
    python evolve_lorenz_ctrnn.py --mode products --z-only # combine flags
"""

import argparse
import math
import multiprocessing
import os
import time
from configparser import ConfigParser
from tempfile import NamedTemporaryFile

import neat

# ---------------------------------------------------------------------------
# Constants (matching the Julia implementation)
# ---------------------------------------------------------------------------

LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0

INTEGRATION_DT = 0.01
SUBSAMPLE = 10
DATA_DT = INTEGRATION_DT * SUBSAMPLE  # 0.1s effective timestep

TOTAL_STEPS = 11000
TRANSIENT_STEPS = 1000
TRAIN_STEPS = 8000
TEST_STEPS = 2000

N_GENERATIONS = 300
PENALTY_FITNESS = -10.0

# ---------------------------------------------------------------------------
# Module-level globals for parallel evaluation
# Set by main() before the ParallelEvaluator pool is created so that forked
# worker processes inherit them.
# ---------------------------------------------------------------------------

_train_inputs = None   # list of lists: input vectors per timestep
_train_targets = None  # list of lists: target vectors per timestep


# ---------------------------------------------------------------------------
# Lorenz system integration (hand-written RK4, no external ODE solver)
# ---------------------------------------------------------------------------

def lorenz_derivatives(x, y, z):
    """Compute dx/dt, dy/dt, dz/dt for the Lorenz system."""
    return (
        LORENZ_SIGMA * (y - x),
        x * (LORENZ_RHO - z) - y,
        x * y - LORENZ_BETA * z,
    )


def lorenz_rk4_step(x, y, z, dt):
    """Advance (x, y, z) by one RK4 step of size dt."""
    k1x, k1y, k1z = lorenz_derivatives(x, y, z)
    k2x, k2y, k2z = lorenz_derivatives(
        x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z
    )
    k3x, k3y, k3z = lorenz_derivatives(
        x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z
    )
    k4x, k4y, k4z = lorenz_derivatives(
        x + dt * k3x, y + dt * k3y, z + dt * k3z
    )
    return (
        x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x),
        y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y),
        z + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z),
    )


def generate_lorenz_trajectory(n_steps, dt, x0=1.0, y0=1.0, z0=1.0):
    """Integrate the Lorenz system for n_steps, returning a list of (x, y, z) tuples."""
    trajectory = []
    x, y, z = x0, y0, z0
    for _ in range(n_steps):
        x, y, z = lorenz_rk4_step(x, y, z, dt)
        trajectory.append((x, y, z))
    return trajectory


# ---------------------------------------------------------------------------
# Normalization to [-1, 1]
# ---------------------------------------------------------------------------

class NormParams:
    """Per-variable min/max for affine normalization to [-1, 1]."""
    def __init__(self, min_vals, max_vals):
        self.min_vals = min_vals
        self.max_vals = max_vals


def compute_norm_params(data):
    """Compute per-variable min and max from a list of (x, y, z) tuples."""
    n_vars = len(data[0])
    min_vals = [min(row[i] for row in data) for i in range(n_vars)]
    max_vals = [max(row[i] for row in data) for i in range(n_vars)]
    return NormParams(min_vals, max_vals)


def normalize_point(point, nrm):
    """Normalize one point to [-1, 1]."""
    result = []
    for i, v in enumerate(point):
        rng = nrm.max_vals[i] - nrm.min_vals[i]
        if rng == 0.0:
            result.append(0.0)
        else:
            result.append(2.0 * (v - nrm.min_vals[i]) / rng - 1.0)
    return result


def normalize_data(data, nrm):
    """Normalize a list of tuples, returning list of lists."""
    return [normalize_point(p, nrm) for p in data]


def denormalize_value(v, var_idx, nrm):
    """Denormalize a single value back to physical units."""
    rng = nrm.max_vals[var_idx] - nrm.min_vals[var_idx]
    return (v + 1.0) / 2.0 * rng + nrm.min_vals[var_idx]


# ---------------------------------------------------------------------------
# Product-augmented inputs
# ---------------------------------------------------------------------------

def augment_with_products(point):
    """Append pairwise products (xy, xz, yz) to a 3-element list.

    Products of values in [-1, 1] remain in [-1, 1], so no extra
    normalization is needed.
    """
    x, y, z = point[0], point[1], point[2]
    return point + [x * y, x * z, y * z]


# ---------------------------------------------------------------------------
# Data preparation pipeline
# ---------------------------------------------------------------------------

def prepare_data(mode='base'):
    """Generate Lorenz trajectory, split into train/test, normalize, and augment.

    Returns (train_input, test_input, norm_params, train_raw, test_raw) where
    train_raw and test_raw are the un-normalized (x, y, z) tuples for
    visualization.
    """
    full_traj = generate_lorenz_trajectory(TOTAL_STEPS, INTEGRATION_DT)

    # Split (subsampled) into transient (discarded), train, and test
    train_start = TRANSIENT_STEPS
    train_end = TRANSIENT_STEPS + TRAIN_STEPS
    test_end = train_end + TEST_STEPS

    train_raw = full_traj[train_start:train_end:SUBSAMPLE]
    test_raw = full_traj[train_end:test_end:SUBSAMPLE]

    # Normalize using training-set statistics only
    norm_params = compute_norm_params(train_raw)
    train_norm = normalize_data(train_raw, norm_params)
    test_norm = normalize_data(test_raw, norm_params)

    # Augment with product terms in "products" mode
    if mode == 'products':
        train_input = [augment_with_products(p) for p in train_norm]
        test_input = [augment_with_products(p) for p in test_norm]
    else:
        train_input = train_norm
        test_input = test_norm

    return train_input, test_input, norm_params, train_raw, test_raw


# ---------------------------------------------------------------------------
# Fitness evaluation (called by ParallelEvaluator in worker processes)
# ---------------------------------------------------------------------------

def eval_genome(genome, config):
    """Evaluate a single genome on the training set.

    Reads module-level _train_inputs and _train_targets (set before the
    multiprocessing pool is forked).

    Returns negative mean squared error (evolution maximizes toward zero).
    """
    net = neat.ctrnn.CTRNN.create(genome, config)
    net.reset()

    n_steps = len(_train_inputs)
    n_outputs = len(_train_targets[0])
    total_se = 0.0

    for t in range(n_steps):
        output = net.advance(_train_inputs[t], DATA_DT, DATA_DT)

        # Penalize genomes that produce NaN, Inf, or very large values.
        # Large finite values occur when per-node time constants are small
        # relative to the integration timestep (dt/tau >> 1), causing the
        # explicit Euler integration to become numerically unstable.
        if any(math.isnan(v) or math.isinf(v) or abs(v) > 1e10 for v in output):
            return PENALTY_FITNESS

        for i in range(n_outputs):
            total_se += (output[i] - _train_targets[t][i]) ** 2

    return -total_se / (n_steps * n_outputs)


# ---------------------------------------------------------------------------
# Test-set evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(winner, config, test_input, target_rows):
    """Run the winner genome on the test set.

    Returns (mse, predictions) where predictions is a list of output vectors
    in normalized space.
    """
    net = neat.ctrnn.CTRNN.create(winner, config)
    net.reset()

    n_steps = len(test_input) - 1
    n_outputs = len(target_rows)
    predictions = []
    total_se = 0.0

    for t in range(n_steps):
        output = net.advance(test_input[t], DATA_DT, DATA_DT)
        predictions.append(list(output))

        for i in range(n_outputs):
            target = test_input[t + 1][target_rows[i]]
            total_se += (output[i] - target) ** 2

    mse = total_se / (n_steps * n_outputs)
    return mse, predictions


# ---------------------------------------------------------------------------
# Pearson correlation
# ---------------------------------------------------------------------------

def pearson_correlation(xs, ys):
    """Compute the Pearson correlation coefficient between two sequences."""
    n = len(xs)
    if n == 0:
        return 0.0
    xm = sum(xs) / n
    ym = sum(ys) / n
    cov = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - xm) ** 2 for x in xs))
    sy = math.sqrt(sum((y - ym) ** 2 for y in ys))
    if sx > 0.0 and sy > 0.0:
        return cov / (sx * sy)
    return 0.0


# ---------------------------------------------------------------------------
# Configuration loading with optional overrides
# ---------------------------------------------------------------------------

def load_config(genome_overrides=None):
    """Load config-ctrnn from this script's directory.

    If genome_overrides is provided, write a temporary config file with the
    modified [DefaultGenome] values and load from that.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config-ctrnn')

    if not genome_overrides:
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path)

    parser = ConfigParser()
    with open(config_path) as f:
        parser.read_file(f)

    for key, value in genome_overrides.items():
        parser.set('DefaultGenome', key, str(value))

    tmp = NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
    try:
        parser.write(tmp)
        tmp.close()
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             tmp.name)
    finally:
        os.unlink(tmp.name)

    return config


# ---------------------------------------------------------------------------
# Visualization (optional, requires matplotlib)
# ---------------------------------------------------------------------------

def try_visualize(winner, config, test_input, test_raw, norm_params,
                  target_rows, target_labels):
    """Generate PNG plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nVisualization skipped (matplotlib not installed).")
        print("  pip install matplotlib")
        return

    print("\nGenerating visualizations...")
    test_mse, pred_norm = evaluate_on_test(winner, config, test_input, target_rows)
    n_pred = len(pred_norm)
    n_outputs = len(target_rows)

    # Denormalize predictions to physical units
    pred_phys = []
    for t in range(n_pred):
        row = []
        for oi, var_idx in enumerate(target_rows):
            row.append(denormalize_value(pred_norm[t][oi], var_idx, norm_params))
        pred_phys.append(row)

    # True test data in physical units (offset by 1 to align with predictions)
    true_phys = test_raw[1:n_pred + 1]

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 3D phase portrait (only for 3-output mode)
    if n_outputs == 3:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'Lorenz: True vs CTRNN Predicted (test MSE={test_mse:.6f})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        tx = [p[0] for p in true_phys]
        ty = [p[1] for p in true_phys]
        tz = [p[2] for p in true_phys]
        ax.plot(tx, ty, tz, color='gray', alpha=0.4, linewidth=1, label='True')

        px = [p[0] for p in pred_phys]
        py = [p[1] for p in pred_phys]
        pz = [p[2] for p in pred_phys]
        ax.plot(px, py, pz, color='orange', linewidth=1.5, label='Predicted')
        ax.legend(loc='upper left')

        path = os.path.join(results_dir, 'lorenz_phase_portrait.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Saved: {path}')

    # Time series comparison
    fig, axes = plt.subplots(n_outputs, 1, figsize=(12, 3 * n_outputs))
    if n_outputs == 1:
        axes = [axes]

    time_axis = [(t + 1) * DATA_DT for t in range(n_pred)]

    for oi, (var_idx, label) in enumerate(zip(target_rows, target_labels)):
        ax = axes[oi]
        true_vals = [p[var_idx] for p in true_phys]
        pred_vals = [p[oi] for p in pred_phys]

        ax.plot(time_axis, true_vals, color='gray', linewidth=1, label='True')
        ax.plot(time_axis, pred_vals, color='orange', linewidth=1.2, label='Predicted')
        ax.set_ylabel(label)
        if oi == n_outputs - 1:
            ax.set_xlabel('Time (s)')
        if oi == 0:
            ax.legend(loc='upper right')

    fig.tight_layout()
    path = os.path.join(results_dir, 'lorenz_time_series.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODE_DESCRIPTIONS = {
    'base':        '3 inputs (x, y, z)',
    'products':    '6 inputs (x, y, z, xy, xz, yz)',
    'product-agg': '3 inputs (x, y, z) + product aggregation',
}


def main():
    global _train_inputs, _train_targets

    parser = argparse.ArgumentParser(
        description='Evolve a CTRNN to predict Lorenz attractor dynamics')
    parser.add_argument('--mode', choices=['base', 'products', 'product-agg'],
                        default='base', help='Input representation mode (default: base)')
    parser.add_argument('--z-only', action='store_true',
                        help='Predict only z instead of all three variables')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()

    mode = args.mode
    z_only = args.z_only
    num_workers = args.workers or multiprocessing.cpu_count()
    start_time = time.time()

    # Determine input/output dimensions
    num_inputs = 6 if mode == 'products' else 3
    target_rows = [2] if z_only else [0, 1, 2]
    num_outputs = len(target_rows)
    target_labels = ['z'] if z_only else ['x', 'y', 'z']

    # Banner
    output_str = 'z only' if z_only else 'x, y, z'
    print('=' * 60)
    print('Lorenz Attractor CTRNN Prediction')
    print('=' * 60)
    print(f'Mode:       {mode} -- {MODE_DESCRIPTIONS[mode]}')
    print(f'Outputs:    {output_str} ({num_outputs})')
    print(f'Lorenz:     sigma={LORENZ_SIGMA}, rho={LORENZ_RHO}, '
          f'beta={LORENZ_BETA:.4f}')
    print(f'Integration: {TOTAL_STEPS} steps at dt={INTEGRATION_DT}, '
          f'subsampled {SUBSAMPLE}x (effective dt={DATA_DT})')
    print(f'Data split: {TRANSIENT_STEPS} transient + {TRAIN_STEPS} train + '
          f'{TEST_STEPS} test integration steps')
    print(f'After subsampling: {TRAIN_STEPS // SUBSAMPLE} train points, '
          f'{TEST_STEPS // SUBSAMPLE} test points')
    print(f'Evolution:  {N_GENERATIONS} generations, pop_size=150')
    print(f'CTRNN:      per-node time constants (evolved)')
    print(f'Workers:    {num_workers}')
    print()

    # --- Prepare data ---
    print('Generating Lorenz trajectory...')
    train_input, test_input, norm_params, train_raw, test_raw = prepare_data(
        mode=mode)
    print(f'  {len(train_input[0])} inputs x {len(train_input)} train points, '
          f'{len(test_input)} test points')
    print('Normalization ranges (training set):')
    for i, label in enumerate(['x', 'y', 'z']):
        print(f'  {label}: [{norm_params.min_vals[i]:.2f}, '
              f'{norm_params.max_vals[i]:.2f}]')
    print()

    # Build input/target arrays for the eval function.
    # Input at step t -> target at step t+1 (next-step prediction).
    n_train_steps = len(train_input) - 1
    _train_inputs = train_input[:n_train_steps]
    _train_targets = [
        [train_input[t + 1][row] for row in target_rows]
        for t in range(n_train_steps)
    ]

    # --- Config overrides for mode ---
    overrides = {}
    if num_inputs != 3:
        overrides['num_inputs'] = num_inputs
    if num_outputs != 3:
        overrides['num_outputs'] = num_outputs
    if mode == 'product-agg':
        overrides['aggregation_options'] = 'sum product'
        overrides['aggregation_mutate_rate'] = '0.1'

    config = load_config(genome_overrides=overrides if overrides else None)

    # --- Create population and reporters ---
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # --- Evolve with parallel evaluation ---
    pe = neat.ParallelEvaluator(num_workers, eval_genome)
    winner = pop.run(pe.evaluate, N_GENERATIONS)

    elapsed = time.time() - start_time

    # --- Results ---
    print()
    print('=' * 60)
    print('Evolution complete')
    print('=' * 60)
    print(f'  Best fitness (train): {winner.fitness:.6f}')
    print(f'  Train MSE:            {-winner.fitness:.6f}')
    print(f'  Nodes:                {len(winner.nodes)}')
    print(f'  Connections:          {len(winner.connections)}')
    print(f'  Wall-clock time:      {elapsed:.1f}s')

    # Evaluate on held-out test set
    test_mse, preds = evaluate_on_test(winner, config, test_input, target_rows)
    print(f'  Test MSE:             {test_mse:.6f}')

    # Per-output Pearson correlation on test set
    n_test_steps = len(preds)
    for oi, (row, label) in enumerate(zip(target_rows, target_labels)):
        pred_vals = [preds[t][oi] for t in range(n_test_steps)]
        true_vals = [test_input[t + 1][row] for t in range(n_test_steps)]
        corr = pearson_correlation(pred_vals, true_vals)
        print(f'  {label} correlation:      {corr:.4f}')
    print()

    # --- Visualization (optional) ---
    try_visualize(winner, config, test_input, test_raw, norm_params,
                  target_rows, target_labels)


if __name__ == '__main__':
    main()
