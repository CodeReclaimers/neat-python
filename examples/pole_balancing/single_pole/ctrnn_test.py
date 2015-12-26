# Test the performance of the best genome produced by ctrnn_evolve.py.
from __future__ import print_function

import pickle

from cart_pole import CartPole, run_simulation, num_steps, discrete_actuator_force
from movie import make_movie

from neat import ctrnn

# load the winner
with open('ctrnn_winner_genome', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

net = ctrnn.create_phenotype(c)
sim = CartPole()

print()
print("Initial conditions:")
print("        x = {0:.4f}".format(sim.x))
print("    x_dot = {0:.4f}".format(sim.dx))
print("    theta = {0:.4f}".format(sim.theta))
print("theta_dot = {0:.4f}".format(sim.dtheta))
print()
n = run_simulation(sim, net, discrete_actuator_force)
print('Pole balanced for {0:d} of {1:d} time steps'.format(n, num_steps))

print()
print("Final conditions:")
print("        x = {0:.4f}".format(sim.x))
print("    x_dot = {0:.4f}".format(sim.dx))
print("    theta = {0:.4f}".format(sim.theta))
print("theta_dot = {0:.4f}".format(sim.dtheta))
print()
print("Making movie...")
make_movie(net, discrete_actuator_force, 15.0, "ctrnn_movie.mp4")