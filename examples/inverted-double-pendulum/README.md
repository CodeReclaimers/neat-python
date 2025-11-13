# Inverted Double Pendulum Example

This example demonstrates using NEAT to evolve a neural network controller for the Gymnasium `InvertedDoublePendulum-v5` environment.

## Problem Description

The inverted double pendulum consists of two poles connected serially and mounted on a cart that moves along a frictionless track. The objective is to balance both poles in an upright position by applying horizontal forces to the cart.

### Environment Details

- **Observation Space**: 9-dimensional continuous vector containing:
  - Position of the cart (x, y, z)
  - Angular positions of both poles (θ1, θ2)
  - Velocities of the cart (ẋ, ẏ, ż)
  - Velocity magnitude at the tip of the second pole
  
- **Action Space**: 1-dimensional continuous action representing the force applied to the cart (ranging from -1 to 1)

- **Reward**: The agent receives a reward for each timestep the poles remain balanced. The episode terminates when the poles fall beyond a certain angle or the maximum number of steps (1000) is reached.

- **Success Criterion**: A fitness of 9000+ (staying balanced for most/all of the maximum 1000 steps over multiple episodes)

## Files

- `evolve-feedforward.py` - Main evolution script using feedforward networks
- `config-feedforward` - NEAT configuration file
- `test-feedforward.py` - Script to test and visualize trained controllers
- `README.md` - This file

## Requirements

```bash
pip install neat-python gymnasium
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
1. Create a population of 150 random neural networks
2. Evolve them for up to 300 generations
3. Use parallel evaluation across all CPU cores
4. Save checkpoints every 10 generations
5. Save the best genome as `winner-feedforward.pickle`
6. Generate visualization files (fitness plots, network diagrams)

The evolution will stop when a genome achieves the fitness threshold (9000.0) or after 300 generations.

### Testing a Trained Controller

To test a trained controller with visualization:

```bash
python test-feedforward.py
```

Or to test a specific genome file:

```bash
python test-feedforward.py path/to/genome.pickle
```

This will run the controller for 5 episodes and display:
- Real-time visualization of the pendulum
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

# Find the most recent checkpoint
pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-XX')
# Continue evolution
# ... (same as in evolve-feedforward.py)
```

## Configuration Notes

The `config-feedforward` file contains several important parameters:

- **Population size**: 150 individuals
- **Network structure**: Starts with no hidden nodes, allowing NEAT to evolve the topology
- **Activation functions**: tanh (default), with sigmoid and relu available through mutation
- **Mutation rates**: Configured to allow both structural and weight mutations
- **Speciation threshold**: 3.0 for maintaining diversity

These parameters can be tuned based on your computational resources and desired performance.

## Tips for Success

1. **Patience**: This is a challenging control problem. Evolution may take many generations to find good solutions.

2. **Parallel evaluation**: The script uses all available CPU cores by default. Adjust the number of workers in `evolve-feedforward.py` if needed.

3. **Hyperparameter tuning**: If evolution stagnates, try:
   - Increasing population size
   - Adjusting mutation rates
   - Changing the activation functions
   - Modifying the compatibility threshold

4. **Multiple runs**: Due to the stochastic nature of evolution, running multiple independent trials may help find better solutions.

## Expected Results

A successfully evolved controller should:
- Balance both poles for the full 1000 steps
- Achieve fitness scores consistently above 9000
- Use relatively small networks (often fewer than 10 nodes)

## References

- [NEAT Paper](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- [Gymnasium Documentation](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/)
- [neat-python Documentation](https://neat-python.readthedocs.io/)
