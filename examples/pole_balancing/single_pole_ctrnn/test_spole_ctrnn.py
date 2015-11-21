# test single pole performance

import os
import random
import cPickle

from neat.config import Config
from neat import ctrnn
import spole_ctrnn


# load the winner
with open('winner_chromosome') as f:
    c = cPickle.load(f)

print 'Loaded chromosome:'
print c

local_dir = os.path.dirname(__file__)
config = Config(os.path.join(local_dir, 'spole_ctrnn_config'))
net = ctrnn.create_phenotype(c)


# initial conditions (as used by Stanley)
x = (random.randint(0, 2 ** 31) % 4800) / 1000.0 - 2.4
x_dot = (random.randint(0, 2 ** 31) % 2000) / 1000.0 - 1
theta = (random.randint(0, 2 ** 31) % 400) / 1000.0 - .2
theta_dot = (random.randint(0, 2 ** 31) % 3000) / 1000.0 - 1.5

print
print "Initial conditions:"
print "        x = %.4f" % x
print "    x_dot = %.4f" % x_dot
print "    theta = %.4f" % theta
print "theta_dot = %.4f" % theta_dot
print

for step in xrange(10 ** 5):

    twelve_degrees = 0.2094384

    # maps into [0,1]   
    inputs = [(x + 2.4) / 4.8,
              (x_dot + 0.75) / 1.5,
              (theta + twelve_degrees) / 0.41,
              (theta_dot + 1.0) / 2.0]

    action = net.pactivate(inputs)

    # Apply action to the simulated cart-pole
    x, x_dot, theta, theta_dot = spole_ctrnn.cart_pole(action[0], x, x_dot, theta, theta_dot)

    if abs(x) >= 2.5 or abs(theta) >= twelve_degrees:
        import sys

        print 'FAILED at step %d' % step
        print "        x = %.4f" % x
        print "    x_dot = %.4f" % x_dot
        print "    theta = %.4f" % theta
        print "theta_dot = %.4f" % theta_dot
        sys.exit(0)

print 'Pole balanced for 10^5 time steps!'
