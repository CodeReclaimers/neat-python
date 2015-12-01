## XOR examples ##

The scripts in this directory show how to evolve networks that implement the 2-input XOR function.  These XOR examples
are intended to be "Hello World" style samples, so that you can see the minimal amount of code you need to write in order
to make use of *neat-python*. 

These simple examples are sometimes also useful as a debugging tool for *neat-python*, because you can step through the 
NEAT-specific code and watch what happens without getting swamped by the complexity of the networks and/or application code.

## The examples ##

* `xor2.py` shows how to evolve a feed-forward neural network with sigmoidal neurons.  This is currently the behavior
you get if you stick with the *neat-python* defaults.

* `xor2_parallel.py` evolves the same type of network as `xor2.py`, but this example shows how you can make use of 
multiple processors to evaluate networks in parallel.  

* `xor2_spiking.py` evolves a network of spiking neurons, using Izhikevich's neuron model from ["Simple model of spiking 
neurons"](http://www.dis.uniroma1.it/~gori/Sito_GG/Modellistica_files/2003%20Net.pdf) in 2003.