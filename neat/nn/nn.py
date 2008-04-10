import random
from math import log, pi, sin
#import psyco; psyco.full()
from nn_cpp import Neuron, Synapse, set_nn_activation
    
class Network(object):
    'A neural network has a list of neurons linked by synapses'
    def __init__(self, neurons=[], links=None):
        self.__neurons = neurons
        self.__synapses = []                
        
        if links is not None:        
            N = {} # a temporary dictionary to create the network connections
            for n in self.__neurons: 
                N[n.id] = n        
            for c in links: 
                self.__synapses.append(Synapse(N[c[0]], N[c[1]], c[2]))
                
    neurons = property(lambda self: self.__neurons)
    synapses = property(lambda self: self.__synapses)
            
    def add_neuron(self, neuron):
        self.__neurons.append(neuron)
        
    def add_synapse(self, synapse):
        self.__synapses.append(synapse)
            
    def __repr__(self):
        return '%d nodes and %d synapses' % (len(self.__neurons), len(self.__synapses))
    
    def sactivate(self, inputs=[]):    
        '''Serial (asynchronous) network activation method. Mostly
           used  in classification tasks (supervised learning) in
           feedforward topologies. All neurons are updated (activated)
           one at a time following their order of importance, so if
           you're defining your own feedforward topology, make sure 
           you got them in the right order of activation.
        '''
        # assign "input neurons'" output values (sensor readings)
        k=0
        for n in self.__neurons[:len(inputs)]:
            if(n.type == 'INPUT'):                
                n.output = inputs[k]
                k+=1
        # activate all neurons in the network (except for the inputs)
        net_output = [] 
        for n in self.__neurons[k:]:
            n.output = n.activate()
            
            if(n.type == 'OUTPUT'): 
                net_output.append(n.output)

        return net_output
        
    def pactivate(self, inputs=[]):
        '''Parallel (synchronous) network activation method. Mostly used
           for control and unsupervised learning (i.e., artificial life) 
           in recurrent networks. All neurons are updated (activated)
           simultaneously.
        '''
        # the current state is like a "photograph" taken at each time step 
        # reresenting all neuron's state at that time (think of it as a clock)
        current_state = []      

        k=0        
        for n in self.__neurons:
            if(n.type == 'INPUT'): 
                n.output = inputs[k]   
                current_state.append(n.output)
                k+=1
            else:
                current_state.append(n.activate())
            
        # updates all neurons at once                
        net_output = []
        for i, n in enumerate(self.__neurons):
            n.output = current_state[i]
            if(n.type == 'OUTPUT'): 
                net_output.append(n.output)
                
        return net_output

def create_phenotype(chromo): 
        """ Receives a chromosome and returns its phenotype (a neural network) """
        
        set_nn_activation('exp')

        #need to figure out how to do it - we need a general enough create_phenotype method
        neurons_list = [Neuron(ng._type, ng._id, ng._bias, ng._response) \
                        for ng in chromo._node_genes]
        
        conn_list = [(cg.innodeid, cg.outnodeid, cg.weight) \
                     for cg in chromo.conn_genes if cg.enabled]        
        
        return Network(neurons_list, conn_list) 
    
def create_ffphenotype(chromo):
    """ Receives a chromosome and returns its phenotype (a neural network) """
    set_nn_activation('exp')
    # first create inputs
    neurons_list = [Neuron('INPUT', ng.id, 0, 0) \
                    for ng in chromo.node_genes if ng.type == 'INPUT']
    
    # Add hidden nodes in the right order
    for id in chromo.node_order:
        neurons_list.append(Neuron('HIDDEN', id, chromo.node_genes[id - 1].bias, chromo.node_genes[id - 1].response))
        
    # finally the output
    neurons_list.extend(Neuron('OUTPUT', ng.id, ng.bias, ng.response) \
                        for ng in chromo.node_genes if ng.type == 'OUTPUT')
    
    assert(len(neurons_list) == len(chromo.node_genes))
    
    conn_list = [(cg.innodeid, cg.outnodeid, cg.weight) \
                 for cg in chromo.conn_genes if cg.enabled] 
    
    return Network(neurons_list, conn_list)        

if __name__ == "__main__":
    # Example    
    set_nn_activation('exp')
    
    # defining a neural network manually    
    neurons = [Neuron('INPUT', 1),  Neuron('INPUT', 2),
               Neuron('HIDDEN', 3, 0.27), 
               Neuron('OUTPUT', 4, 0.27)]       
    connections = [(1, 3, 0.75), (2, 3, -0.2), (3, 4, -0.1), (4, 3, 0.25)] 
    
    net = Network(neurons, connections) # constructs the neural network
    print net # print how many neurons and synapses our network has 
    
    for i in xrange(10):
        print net.pactivate([0.1, 0.1])  # parallel activation method
    
    

