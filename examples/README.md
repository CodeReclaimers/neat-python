## neat-python examples ##

Note that there is a significant amount of duplication between these scripts, and this is intentional.  The goal is to 
make it easier to see what the example is doing, without making the user dig through a bunch of utility code that is not 
directly related to the NEAT library usage.

Documentation for the library and some of the examples may be found [here](http://neat-python.readthedocs.io/en/latest/).

## The examples ##

* `xor` A "hello world" sample showing basic usage.

* `circuits` Uses an external circuit simulator (PySpice) to create electronic circuits that reproduce an arbitrary function of the input voltage.

* `memory-fixed` Reproduce a fixed-length sequence of binary inputs.    

* `memory-variable` Reproduce a variable-length sequence of binary inputs.

* `neuron-demo` Plot outputs of very simple CTRNN and Izhikevich networks.
 
* `openai-lander` Solve the [OpenAI Gym LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2) environment.

* `picture2d` Generate 2D color or grayscale images using Compositional Pattern-Producing Networks.

* `single-pole-balancing` Balance a pole on top of a movable cart.