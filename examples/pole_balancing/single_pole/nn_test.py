# Test the performance of the best genome produced by nn_evolve.py.
import os
import cPickle

from neat import nn
from neat.config import Config
from cart_pole import CartPole, run_simulation, num_steps, discrete_actuator_force
from movie import make_movie


# load the winner
with open('nn_winner_genome') as f:
    c = cPickle.load(f)

print 'Loaded chromosome:'
print c

local_dir = os.path.dirname(__file__)
config = Config(os.path.join(local_dir, 'nn_config'))
net = nn.create_phenotype(c)

sim = CartPole()

print
print "Initial conditions:"
print "        x = %.4f" % sim.x
print "    x_dot = %.4f" % sim.dx
print "    theta = %.4f" % sim.theta
print "theta_dot = %.4f" % sim.dtheta
print

n = run_simulation(sim, net, discrete_actuator_force)
print 'Pole balanced for %d of %d time steps' % (n, num_steps)

print
print "Final conditions:"
print "        x = %.4f" % sim.x
print "    x_dot = %.4f" % sim.dx
print "    theta = %.4f" % sim.theta
print "theta_dot = %.4f" % sim.dtheta
print

print "Making movie..."
make_movie(net, discrete_actuator_force, 15.0, "nn_movie.mp4")