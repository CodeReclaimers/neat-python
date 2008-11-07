try:
    import ann # C++ extension
except ImportError:
    print "Neural network extension library not found!"
    raise

def create_ffphenotype(chromo):
    """ Receives a chromosome and returns its phenotype (a neural network) """

    num_inputs =  chromo.sensors
    num_neurons =  len(chromo.node_genes) - num_inputs
    num_outputs = chromo.actuators
    network = ann.ANN(num_inputs, num_neurons)

    if chromo.node_genes[-1].activation_type == 'tanh':
        network.set_logistic(0)

    # creates a dict mapping node_order + output node to [0, 1, 3, ... , n]
    value = 0
    mapping = {}

    # hidden nodes
    for id in  chromo.node_order:
        mapping[id] = value
        #print 'Hidden params: ',value, chromo.node_genes[id - 1].bias, chromo.node_genes[id - 1].response, 0
        network.set_neuron(value, chromo.node_genes[id - 1].bias, chromo.node_genes[id - 1].response, 0)
        value += 1
    # output nodes
    for ng in chromo.node_genes[num_inputs:num_outputs+num_inputs]:
        mapping[ng.id] = value
        #print 'Output params: ', value, ng.bias, ng.response, 1
        network.set_neuron(value, ng.bias, ng.response, 1)
        value += 1

    for cg in chromo.conn_genes:
        if cg.enabled:
            if cg.innodeid-1 < num_inputs:
                # set sensory input
                network.set_sensory_weight(cg.innodeid-1, mapping[cg.outnodeid], cg.weight)
                #print "Sensory: ", cg.innodeid-1, mapping[cg.outnodeid], cg.weight
            else:
                # set interneuron connection
                network.set_synapse(mapping[cg.innodeid], mapping[cg.outnodeid], cg.weight)
                #print "Inter..: ", mapping[cg.innodeid], mapping[cg.outnodeid], cg.weight

    return network

def create_phenotype(chromo):
    num_inputs  =  chromo.sensors
    num_neurons =  len(chromo.node_genes) - num_inputs
    #num_outputs = chromo.actuators

    network = ann.ANN(num_inputs, num_neurons)

    if chromo.node_genes[-1].activation_type == 'tanh':
        network.set_logistic(0)

    # create neurons
    neuron_type = None
    for ng in chromo.node_genes[num_inputs:]:
        if ng.type == 'OUTPUT':
            neuron_type = 1
        else:
            neuron_type = 0
        #print 'Creating neuron: ', ng.id-num_inputs-1, ng.bias, ng.response, neuron_type
        network.set_neuron(ng.id-num_inputs-1, ng.bias, ng.response, neuron_type)

    # create connections
    for cg in chromo.conn_genes:
        if cg.enabled:
            if cg.innodeid-1 < num_inputs:
                # set sensory input
                network.set_sensory_weight(cg.innodeid-1, cg.outnodeid-num_inputs-1, cg.weight)
                #print "Sensory: ", cg.innodeid-1, cg.outnodeid-num_inputs-1, cg.weight
            else:
                # set interneuron connection
                network.set_synapse(cg.innodeid-num_inputs-1, cg.outnodeid-num_inputs-1, cg.weight)
                #print "Inter..: ", cg.innodeid-num_inputs, cg.outnodeid-num_inputs-1, cg.weight

    return network

if __name__ == "__main__":
    # setting a network manually
    network = ann.ANN(2,2)
    #network.set_logistic(True)
    network.set_neuron(0, 0.0, 4.924273, 0)
    network.set_neuron(1, 0.0, 4.924273, 1)

    network.set_sensory_weight(1, 1, -0.09569)
    network.set_sensory_weight(0, 0, 1.0)
    network.set_synapse(0,1,0.97627)

    for i in range(10):
        print network.pactivate([1.0, 0.0])
