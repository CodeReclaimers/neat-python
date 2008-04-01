from math import log, pi, sin
import random
from neat.config import Config

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
        
        if Config.nn_activation == 'exp' or Config.nn_activation == 'tanh':
            set_nn_activation(Config.nn_activation)
        else:
            raise Exception, 'Invalid activation type selected: ' + str(Config.nn_activation)
                
    neurons = property(lambda self: self.__neurons)
    synapses = property(lambda self: self.__synapses)
            
    def add_neuron(self, neuron):
        self.__neurons.append(neuron)
        
    def add_synapse(self, synapse):
        self.__synapses.append(synapse)
            
    def __repr__(self):
        return '%d nodes and %d synapses' % (len(self.__neurons), len(self.__synapses))
    
    def activate(self, inputs=[]):
        if Config.feedforward:
            return self.sactivate(inputs)
        else:
            return self.pactivate(inputs)

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
            if(n._type == 'INPUT'):                
                n._output = inputs[k]
                k+=1
        # activate all neurons in the network (except for the inputs)
        net_output = [] 
        for n in self.__neurons[k:]:
            n._output = n.activate()
            
            if(n._type == 'OUTPUT'): 
                net_output.append(n._output)

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

class FeedForward(Network):
    'A feedforward network is a particular class of neural network'
    # only one hidden layer is considered for now
    def __init__(self, layers, use_bias=False):
        super(FeedForward, self).__init__()
        
        self.__input_layer   = layers[0]
        self.__output_layer  = layers[-1]
        self.__hidden_layers = layers[1:-1]
        self.__use_bias = use_bias
        
        self.__create_net()
        
    def __create_net(self):
        
        # assign random weights for bias
        if self.__use_bias:
            r = random.uniform
        else:
            r = lambda a,b: 0
        
        for i in xrange(self.__input_layer):
            self.add_neuron(Neuron('INPUT'))                              
        
        for i in xrange(self.__hidden_layers[0]):
            self.add_neuron(Neuron('HIDDEN', bias = r(-1,1), response = 1))
            
        for i in xrange(self.__output_layer):
            self.add_neuron(Neuron('OUTPUT', bias = r(-1,1), response = 1))
           
        r = random.uniform  # assign random weights             
        # inputs -> hidden
        for i in self.neurons[:self.__input_layer]:
                for h in self.neurons[self.__input_layer:-self.__output_layer]:
                        self.add_synapse(Synapse(i, h, r(-1,1)))       
        # hidden -> outputs
        for h in self.neurons[self.__input_layer:-self.__output_layer]:
                for o in self.neurons[-self.__output_layer:]:
                        self.add_synapse(Synapse(h, o, r(-1,1)))

def create_phenotype(chromo): 
        """ Receives a chromosome and returns its phenotype (a neural network) """

        #need to figure out how to do it - we need a general enough create_phenotype method
        neurons_list = [Neuron(ng._type, ng._id, ng._bias, ng._response) \
                        for ng in chromo._node_genes]
        
        conn_list = [(cg.innodeid, cg.outnodeid, cg.weight) \
                     for cg in chromo.conn_genes if cg.enabled] 
        
        return Network(neurons_list, conn_list) 
    
def create_ffphenotype(chromo):
    """ Receives a chromosome and returns its phenotype (a neural network) """
    
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
    from neat import visualize
    Config.nn_activation = 'exp'
    
    nn = FeedForward([2,2,1], False)
    ##visualize.draw_ff(nn)
    print 'Serial activation method: '
    for t in range(3):
        print nn.sactivate([1,1])

    #print 'Parallel activation method: '
    #for t in range(3):
        #print nn.pactivate([1,1])

    #print nn

    # defining a neural network manually
    #neurons = [Neuron('INPUT', 1), Neuron('HIDDEN', 2), Neuron('OUTPUT', 3)]
    #connections = [(1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.5)]
    
    #net = Network(neurons, connections) # constructs the neural network
    ###visualize.draw_ff(net)
    #print net.pactivate([0.04]) # parallel activation method
    #print net # print how many neurons and synapses our network has 
    
    ## CTRNN neurontype, id = None, bias, response=1.0, state, tau
#    n1 = CTNeuron('OUTPUT', 1, -2.75, 1.0, -0.0840006, 1.0)
#    n2 = CTNeuron('OUTPUT', 2, -1.75, 1.0, -0.4080350, 1.0)
#    ##n3 = CTNeuron('OUTPUT', +0.4000000, 1.0, -0.40)
#
#    connections = [(1, 1, +4.5), (1, 2, -1.0),
#                   (2, 1, +1.0), (2, 2, +4.5)]
#    
#    ##connections = [(1, 1, +4.5), (1, 2, -1.0), (1, 3, +2.3),
#                   ##(2, 1, +1.0), (2, 2, +4.5), (2, 3, +0.8),
#                   ##(3, 1, -1.5), (3, 2, +2.1), (3, 3, +3.1)]
#                   
#    net = Network([n1, n2], connections)
#    
#    ###print net.synapses
#    ###visualize.draw_ff(net)
#    ### we need a method to print NN's current state (as is the case with CTRNNs)
#    for i in xrange(5500):
#        net_output = net.pactivate([0.1])  #forwards one time-step
#        print "%s %s" %(net_output[0], net_output[1])
        ###print "%f " %(net_output[0])
        
        ##print n1.output, n2.output, n3.output # same as n1.current_output
