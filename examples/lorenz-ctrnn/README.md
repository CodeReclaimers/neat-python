# Lorenz Attractor CTRNN Prediction

Evolves a Continuous-Time Recurrent Neural Network (CTRNN) using NEAT to predict the next-step state of the Lorenz attractor. The Lorenz system is a chaotic dynamical system where small errors grow exponentially, making it a challenging benchmark for evolved networks.

Ported from the Julia [NeatEvolution](https://github.com/alanderos91/NeatEvolution.jl) lorenz_ctrnn example.

## Running

```bash
# Default: 3 inputs (x,y,z), 3 outputs (x,y,z), sum aggregation
python examples/lorenz-ctrnn/evolve_lorenz_ctrnn.py

# Pre-computed product inputs: 6 inputs (x,y,z,xy,xz,yz)
python examples/lorenz-ctrnn/evolve_lorenz_ctrnn.py --mode products

# Product aggregation: 3 inputs, but nodes can use * instead of +
python examples/lorenz-ctrnn/evolve_lorenz_ctrnn.py --mode product-agg

# Any mode can be combined with --z-only to predict only z
python examples/lorenz-ctrnn/evolve_lorenz_ctrnn.py --mode product-agg --z-only

# Control parallelism (default: all CPUs)
python examples/lorenz-ctrnn/evolve_lorenz_ctrnn.py --workers 4
```

Runs 300 generations with a population of 150, using parallel fitness evaluation across all available CPU cores.

## Input modes

| Mode | Inputs | Description |
|------|--------|-------------|
| `base` (default) | x, y, z | Standard 3-variable state |
| `products` | x, y, z, xy, xz, yz | Pre-computed pairwise products |
| `product-agg` | x, y, z | Same inputs as base, but `product` added as an aggregation function alongside `sum`, so evolution can discover multiplicative nodes |

The three modes test different hypotheses about why z prediction fails: is the problem that the network lacks access to product terms (`products` mode), or that it lacks the ability to represent multiplication internally (`product-agg` mode)?

## How it works

The Lorenz system is defined by three coupled ODEs:

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

with standard parameters sigma=10, rho=28, beta=8/3. The system is integrated with a hand-written 4th-order Runge-Kutta method (no external ODE solver dependency).

A trajectory of 11,000 steps (dt=0.01) is generated and subsampled every 10th step, giving an effective data timestep of 0.1s. The first 1,000 integration steps are discarded as transient. The next 8,000 form the training set (800 data points after subsampling), and the final 2,000 are held out for testing (200 data points).

Each CTRNN receives the normalized current state as input and predicts the normalized next-step state (or just z). Fitness is negative mean squared error, so evolution drives it toward zero. The CTRNN's continuous-time dynamics make it naturally suited to modeling temporal systems.

## Differences from the Julia version

The Julia NeatEvolution CTRNN implementation evolves **per-node time constants** as part of the genome, with each node having its own time constant in the range [0.01, 5.0]. This allows different nodes to operate at different timescales -- some responding quickly (small tau) and others integrating slowly (large tau).

neat-python's CTRNN implementation uses a **single fixed time constant** for all nodes, set at network creation time. This example uses tau=1.0 (matching the Julia version's initialization mean). This is a genuine capability difference: the Python CTRNN cannot learn different timescales for different nodes.

Additionally, the Julia implementation separates disjoint and excess gene coefficients for the genetic distance calculation, while neat-python combines them into a single `compatibility_disjoint_coefficient`.

## Visualization (optional)

If matplotlib is installed, the script generates PNG plots in `results/`:

- **lorenz_phase_portrait.png** -- 3D phase portrait (3-output mode only)
- **lorenz_time_series.png** -- Per-variable time series on the test set

```bash
pip install matplotlib
```

## Configuration highlights

- **3 or 6 inputs** (with `--mode products`), **3 or 1 outputs** (with `--z-only`)
- **No initial hidden nodes** -- NEAT discovers the topology
- **feed_forward = False** -- required for CTRNN recurrence
- **time_constant = 1.0** -- fixed for all nodes (see differences above)
- **bias/weight range [-5, 5]** -- narrower bounds for tanh activation
- **max_stagnation = 30** -- patient stagnation threshold for a harder task
- **10x subsampling** -- ensures per-step state change is large enough that networks must learn dynamics, not just copy input
- **product-agg mode** -- adds `product` to aggregation options with 10% mutation rate
