import os
import sys
import cPickle

from neat.config import Config
from cart_pole import CartPole

filename = 'winner_chromosome'
if len(sys.argv) > 1:
    filename = sys.argv[1]

# load genome
print "loading genome {0!s}".format(filename)
with open(filename) as f:
    c = cPickle.load(f)

# load settings file
local_dir = os.path.dirname(__file__)
config = Config(os.path.join(local_dir, 'dpole_config'))

print "Loaded genome:\n{0!s}".format(c)
# starts the simulation
simulator = CartPole([c], markov=False)
simulator.run(testing=True)
