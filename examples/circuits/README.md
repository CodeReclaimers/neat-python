## Circuit example ##

NOTE: This example is still fairly rough around the edges.  The path to my ngspice implementation
is hard-coded, and I eventually want to have something in the circuit other than diodes and 
resistors.  

The scripts in this directory show how to evolve a network that describes an electronic 
 circuit that reproduces an arbitrary function of the input voltage.

Note that there is a significant amount of duplication between these scripts, and this is intentional.  The goal is to 
make it easier to see what the example is doing, without making the user dig through a bunch of code that is not 
directly related to the NEAT library usage.
