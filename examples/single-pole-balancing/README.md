## Single-pole balancing examples ##

The scripts in this directory show how to evolve a network that balances a pole on top of a movable cart.  The network
inputs are the cart position, pole angle, and the derivatives of cart position and pole angle.  The network output is 
used to apply a force to the cart in either the left or right direction.

Note that there is a significant amount of duplication between these scripts, and this is intentional.  The goal is to 
make it easier to see what the example is doing, without making the user dig through a bunch of code that is not 
directly related to the NEAT library usage.

## Cart/pole system dynamics ##

The system dynamics are taken from ["Correct equations for the dynamics of the cart-pole system"](http://coneural.org/florian/papers/05_cart_pole.pdf)
by [Razvan V. Florian](http://florian.io).  Numerical integration is done by the [leapfrog method](https://en.wikipedia.org/wiki/Leapfrog_integration).
       


## Running the examples ##

* Run `evolve-feedforward.py`.  When it completes, it will have created the following output:
    - `avg_fitness.svg` Plot of mean/max fitness vs. generation.
    - `speciation.svg` Speciation plot vs. generation.
    - `winner-feedforward` A pickled version of the most fit genome.
    - `winner-feedforward.gv` Graphviz layout of the full winning network.
    - `winner-feedforward.gv.svg` Rendered image of the full winning network.
    - `winner-feedforward-enabled.gv` Graphviz layout of the winning network, with disabled connections removed.
    - `winner-feedforward-enabled.gv.svg` Rendered image of the winning network, with disabled connections removed.
    - `winner-feedforward-enabled-pruned.gv` Graphviz layout of the winning network, with disabled and non-functional connections removed.
    - `winner-feedforward-enabled-pruned.gv.svg` Rendered image of the winning network, with disabled and non-functional connections removed.
    
* Run `feedforward-test.py`.  It will load the most fit genome from `winner-feedforward` and run it in a new simulation to test its
performance.  It will also run a second simulation, and produce a movie `feedforward-movie.mp4` showing the behavior.  (See a sample
movie [here](http://gfycat.com/CavernousCheeryIbadanmalimbe).)

