# test single pole performance
from neat import config, chromosome, genome2
import random, sys
import cPickle as pickle
from cart_pole import CartPole

if len(sys.argv) > 1:
    # load genome
    try: 
        file = open(sys.argv[1], 'r')
    except IOError:
        print "Filename: '"+sys.argv[1]+"' not found!"
        sys.exit(0)
    else:
        c = pickle.load(file)
        file.close()
else:
    print "Loading default winner chromosome file"
    try: 
        file = open('winner_chromosome', 'r')
    except IOError:
        print "Winner chromosome not found!"
        sys.exit(0)
    else:
        c = pickle.load(file)
        file.close()
    
# load settings file
config.load('dpole_config_ctrnn')
# set node gene type
chromosome.node_gene_type = genome2.CTNodeGene
print "Loaded genome:"
print c
# starts the simulation
simulator = CartPole([c], markov=False)             
simulator.run(testing=True)
    
