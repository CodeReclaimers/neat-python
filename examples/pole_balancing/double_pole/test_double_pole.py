import os
import sys
import cPickle as pickle

from neat.config import Config
from cart_pole import CartPole

filename = 'winner_chromosome'
if len(sys.argv) > 1:
    filename = sys.argv[1]

# load genome
print "loading genome %s" % filename
with open(filename) as f:
    c = pickle.load(f)

# load settings file
local_dir = os.path.dirname(__file__)
config = Config(os.path.join(local_dir, 'dpole_config'))

print "Loaded genome:\n%s" % c
# starts the simulation
simulator = CartPole([c], markov=False)
simulator.run(testing=True)
