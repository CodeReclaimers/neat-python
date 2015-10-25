# Test the performance of the genome produced by single_pole.py.

import sys
from random import randint
import cPickle as pickle

from neat.nn import nn_pure as nn
from neat.config import Config
from evolve_single_pole import cart_pole, angle_limit


# load the winner
with open('winner_chromosome') as f:
    c = pickle.load(f)

print 'Loaded chromosome:'
print c

config = Config('spole_config')
net = nn.create_phenotype(c)

# initial conditions (as used by Stanley)
x = randint(0, 4799) / 1000.0 - 2.4
x_dot = randint(0, 1999) / 1000.0 - 1.0
theta = randint(0, 399) / 1000.0 - 0.2
theta_dot = randint(0, 2999) / 1000.0 - 1.5

print "\nInitial conditions:"
print "%2.4f   %2.4f   %2.4f   %2.4f" % (x, x_dot, theta, theta_dot)
for step in xrange(10 ** 5):
    # maps into [0,1]
    inputs = [(x + 2.4) / 4.8,
              (x_dot + 0.75) / 1.5,
              (theta + angle_limit) / 0.41,
              (theta_dot + 1.0) / 2.0]

    action = net.pactivate(inputs)

    # Apply action to the simulated cart-pole
    x, x_dot, theta, theta_dot = cart_pole(action[0], x, x_dot, theta, theta_dot)

    if abs(x) >= 2.4 or abs(theta) >= angle_limit:
        print '\nFailed at step %d \n' % step
        sys.exit(1)

print '\nPole balanced for 10^5 time steps!'
