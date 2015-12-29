# Test the performance of the best genome produced by nn_evolve.py.

# Test the performance of the best genome produced by nn_evolve.py.
from __future__ import print_function

import pickle

from cart_pole import CartPole, discrete_actuator_force
from movie import make_movie

from neat import nn

# load the winner
with open('nn_winner_genome', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

net = nn.create_feed_forward_phenotype(c)
sim = CartPole()

print()
print("Initial conditions:")
print("        x = {0:.4f}".format(sim.x))
print("    x_dot = {0:.4f}".format(sim.dx))
print("    theta = {0:.4f}".format(sim.theta))
print("theta_dot = {0:.4f}".format(sim.dtheta))
print()

# Run the given simulation for up to 100k time steps.
num_balanced = 0
for s in range(10 ** 5):
    inputs = sim.get_scaled_state()
    action = net.serial_activate(inputs)

    # Apply action to the simulated cart-pole
    force = discrete_actuator_force(action)
    sim.step(force)

    # Stop if the network fails to keep the cart within the position or angle limits.
    # The per-run fitness is the number of time steps the network can balance the pole
    # without exceeding these limits.
    if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
        break

    num_balanced += 1


print('Pole balanced for {0:d} of 100k time steps'.format(num_balanced))

print()
print("Final conditions:")
print("        x = {0:.4f}".format(sim.x))
print("    x_dot = {0:.4f}".format(sim.dx))
print("    theta = {0:.4f}".format(sim.theta))
print("theta_dot = {0:.4f}".format(sim.dtheta))
print()
print("Making movie...")
make_movie(net, discrete_actuator_force, 15.0, "nn_movie.mp4")