# Hopper MuJoCo Example

This example demonstrates using NEAT to evolve a neural network controller for the Gymnasium `Hopper-v5` environment.

## Problem Description

The Hopper is a two-dimensional one-legged robot with four main body parts: torso, thigh, leg, and foot. The goal is to hop forward as far as possible without falling by applying torques at the three hinge joints.

### Environment Details

- **Observation Space**: 11-dimensional continuous vector containing joint positions and velocities for the torso and leg segments, plus the vertical position of the torso.
- **Action Space**: 3-dimensional continuous action representing the torques applied at the thigh, leg, and foot joints (each in the range [-1, 1]).
- **Reward**: At each timestep the agent receives a reward that combines forward progress, a survival bonus while the hopper remains healthy, and a small penalty on large control torques.
- **Episode Length**: Up to 1000 timesteps.
- **Success Criterion**: An average fitness (episode return) of roughly **3800+** over multiple evaluation episodes is a good indicator of a strong policy.

## Files

- `evolve-feedforward.py` - Main evolution script using feedforward networks
- `config-feedforward` - NEAT configuration file
- `test-feedforward.py` - Script to test and visualize trained controllers
- `check_dependencies.py` - Helper script to verify required packages and environment
- `clean.sh` - Utility script to remove generated artifacts (checkpoints, plots, etc.)
- `README.md` - This file

## Requirements

The easiest way to install all example dependencies is to use the `examples/requirements.txt` file from the repository root:

```bash
pip install -r examples/requirements.txt
```

This will install `gymnasium[box2d,mujoco]`, `numpy`, `matplotlib`, and other optional packages used by multiple examples.

If you prefer to install only the minimal dependencies for this example, you can run:

```bash
pip install neat-python gymnasium[mujoco]
```

For visualization of the environment, you may also need:

```bash
pip install pygame
```

## Usage

### Training a Controller

To evolve a controller from scratch:

```bash
python evolve-feedforward.py
```

This will:

1. Create a population of random neural networks
2. Evolve them for up to 300 generations (or until the fitness threshold is reached)
3. Use parallel evaluation across all CPU cores
4. Save checkpoints every 10 generations
5. Save the best genome as `winner-feedforward.pickle`
6. Generate visualization files (fitness plots, network diagrams)

The evolution will stop when a genome achieves the fitness threshold (3800.0) or after 300 generations.

### Testing a Trained Controller

To test a trained controller with visualization:

```bash
python test-feedforward.py
```

Or to test a specific genome file:

```bash
python test-feedforward.py path/to/genome.pickle
```

This will run the controller for several episodes and display:

- Real-time visualization of the hopper
- Step count and fitness for each episode
- Average, max, and min fitness across episodes

### Resuming from a Checkpoint

If training is interrupted, you can resume from the most recent checkpoint:

```python
import neat
import os

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Restore from a specific checkpoint
pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-XX')

# Continue evolution (same as in evolve-feedforward.py)
# winner = pop.run(fitness_function, n_generations)
```

## Configuration Notes

The `config-feedforward` file contains several important parameters:

- **Population size**: 150 individuals
- **Fitness threshold**: 3800.0 (approximate "solved" score for Hopper-v5)
- **Network structure**: Starts with no hidden nodes, allowing NEAT to evolve the topology
- **Activation functions**: `tanh` (default), with `sigmoid` and `relu` available through mutation
- **Mutation rates**: Configured to allow both structural and weight mutations
- **Speciation threshold**: 3.0 for maintaining diversity

You can tune these parameters based on your computational resources and desired performance.

## Tips for Success

1. **Patience**: Hopper is a moderately challenging continuous control problem. Evolution may take many generations to find robust solutions.
2. **Parallel evaluation**: The script uses all available CPU cores by default. Adjust the number of workers in `evolve-feedforward.py` if needed.
3. **Hyperparameter tuning**: If evolution stagnates, try:
   - Increasing population size
   - Adjusting mutation rates
   - Changing the activation functions
   - Modifying the compatibility threshold
4. **Multiple runs**: Due to the stochastic nature of evolution, running multiple independent trials may help find better solutions.

## Expected Results

A successfully evolved controller should:

- Learn to hop forward without falling for most of the 1000-step episode
- Achieve fitness scores consistently above 3800
- Use relatively compact networks (often with a small number of hidden nodes)

## References

- Original NEAT paper: Stanley & Miikkulainen, "Evolving Neural Networks through Augmenting Topologies"
- Gymnasium Hopper environment documentation
- neat-python documentation
