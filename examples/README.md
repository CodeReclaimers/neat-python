## neat-python examples ##

Note that there is a significant amount of duplication between these scripts, and this is intentional.  The goal is to 
make it easier to see what the example is doing, without making the user dig through a bunch of utility code that is not 
directly related to the NEAT library usage.

Documentation for the library and some of the examples may be found [here](http://neat-python.readthedocs.io/en/latest/).

For convenience, a conda environment that supports running all of the examples is available in this directory (see `environment.yml`).

## The examples ##

* `xor` A "hello world" sample showing basic usage on the 2-input XOR problem.

* `memory-fixed` Reproduce a fixed-length sequence of binary inputs (currently experimental / not working as intended).

* `memory-variable` Reproduce a variable-length sequence of binary inputs (currently experimental / not working as intended).

* `neuron-demo` Plot outputs of very simple CTRNN and Izhikevich neuron models.

* `single-pole-balancing` Balance a pole on top of a movable cart and optionally render videos of successful policies.

* `picture2d` Generate 2D color or grayscale images using Compositional Pattern-Producing Networks, including an interactive picture-breeder style demo.

* `lunar-lander` Solve the [Gymnasium LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment with a modern API-compatible example.

* `bipedal-walker` Control the [Gymnasium BipedalWalker-v3](https://gymnasium.farama.org/environments/box2d/bipedal_walker/) environment using a continuous-action NEAT policy.

* `inverted-double-pendulum` Evolve a controller for the [Gymnasium InvertedDoublePendulum-v5](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/) environment using a MuJoCo-based simulator.

* `hopper` Evolve a controller for the [Gymnasium Hopper-v5](https://gymnasium.farama.org/environments/mujoco/hopper/) environment using a MuJoCo-based continuous-control task.

* `parallel-reproducible` Demonstrate how to use `ParallelEvaluator` with deterministic seeding to get reproducible parallel evolution runs.

* `export` Train a small XOR network and export the resulting NEAT network to a framework-agnostic JSON format.

* `openai-lander` Archived OpenAI Gym LunarLander example containing pre-trained winner genomes referenced in the documentation.
