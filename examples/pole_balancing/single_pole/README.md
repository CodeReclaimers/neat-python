## Single-pole balancing examples ##

The scripts in this directory show how to evolve a network that balances a pole on top of a movable cart.  The network
inputs are the cart position, pole angle, and the derivatives of cart position and pole angle.  The network output is 
used to apply a constant-magnitude force to the cart in either the left or right direction.

Each set of scripts is named with a prefix to indicate the network type used: *nn_* for a discrete-time recurrent network, and
*ctrnn_* for a continuous-time recurrent network.  

## Running the examples ##

These instructions reference the discrete-time sample; to run the continuous-time version, simply run the *ctrnn_* scripts instead. 

* Run `nn_evolve.py`.  When it completes, it will have created the following output:
    - `nn_fitness.svg`: Plot of mean/max fitness vs. generation.
    - `nn_speciation.svg`: Speciation plot vs. generation.
    - `nn_winner.gv` and `nn_winner.gv.svg`: Graphviz file and rendered image of the network.
    - `nn_winner_genome`: A pickled version of the most fit genome.
    
* Run `nn_test.py`.  It will load the most fit genome from `nn_winner_genome` and run it in a new simulation to test its
performance.  It will also run a second simulation, and produce a movie `nn_movie.mp4` showing the behavior.  (See a sample
movie [here](http://gfycat.com/CavernousCheeryIbadanmalimbe).)

