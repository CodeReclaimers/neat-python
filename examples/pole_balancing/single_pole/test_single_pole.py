# Test the performance of the genome produced by single_pole.py.

import os
import sys
from random import randint
import cPickle

from neat.nn import nn_pure as nn
from neat.config import Config
from cart_pole import CartPole, angle_limit_radians


# load the winner
with open('winner_chromosome') as f:
    c = cPickle.load(f)

print 'Loaded chromosome:'
print c

local_dir = os.path.dirname(__file__)
config = Config(os.path.join(local_dir, 'spole_config'))
net = nn.create_phenotype(c)

sim = CartPole()

print "\nInitial conditions:"
print "%2.4f   %2.4f   %2.4f   %2.4f" % (sim.x, sim.dx, sim.theta, sim.dtheta)
for step in xrange(10 ** 5):
    # maps into [0,1]
    inputs = sim.get_scaled_state()
    action = net.pactivate(inputs)
    force = action[0]

    # Apply action to the simulated cart-pole
    sim.step(force)

    if abs(sim.x) >= 2.4 or abs(sim.theta) >= angle_limit_radians:
        print '\nFailed at step %d \n' % step
        sys.exit(1)

print '\nPole balanced for 10^5 time steps!'
