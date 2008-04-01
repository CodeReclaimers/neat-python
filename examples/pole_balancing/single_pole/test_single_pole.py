# test single pole performance

from neat import config, chromosome, genome2
from neat import nn
from random import randint
import cPickle as pickle
import single_pole

chromosome.node_gene_type = genome2.NodeGene

# load the winner
file = open('winner_chromosome', 'r')
c = pickle.load(file)
file.close()

print 'Loaded chromosome:'
print c

config.load('spole_config')
net = nn.create_phenotype(c)


#x = 0.0
#x_dot = 0.0
#theta = 0.0
#theta_dot = 0.0

# initial conditions (as used by Stanley)
x         = randint(0, 4799)/1000.0 - 2.4
x_dot     = randint(0, 1999)/1000.0 - 1.0
theta     = randint(0,  399)/1000.0 - 0.2
theta_dot = randint(0, 2999)/1000.0 - 1.5
        
print "\nInitial conditions:"
print "%2.4f   %2.4f   %2.4f   %2.4f" %(x, x_dot, theta, theta_dot)
for step in xrange(10**5):

    twelve_degrees=0.2094384
    
    # maps into [0,1]
    inputs = [(x + 2.4)/4.8, 
              (x_dot + 0.75)/1.5,
              (theta + twelve_degrees)/0.41,
              (theta_dot + 1.0)/2.0]
              
    action = net.pactivate(inputs)
    
    # Apply action to the simulated cart-pole
    x, x_dot, theta, theta_dot = single_pole.cart_pole(action[0], x, x_dot, theta, theta_dot)
    
    if (abs(x) >= 2.4 or abs(theta) >= twelve_degrees):
    #if abs(theta) >= twelve_degrees: # Igel (p. 5)
        import sys
        sys.stderr.write('\nFailed at step %d \n' %step)
        sys.exit(0)
    
print '\nPole balanced for 10^5 time steps!'
           
    #print "%f \t %f \t %f" %(action[0], x, theta)
    #if (x < -2.4 or x > 2.4 or theta < -twelve_degrees or theta > twelve_degrees):
    
