try:
    import ctrnn # C++ extension
except ImportError:
    print "CTRNN extension library not found!"
    raise

def create_phenotype(chromo):
    num_inputs =  chromo.sensors
    num_neurons =  len(chromo.node_genes) - num_inputs
    #num_outputs = chromo.actuators

    network = ctrnn.CTRNN(num_inputs, num_neurons)
    #network.set_rk4(0.01) # integration method
    network.set_euler(0.01)

    if chromo.node_genes[-1].activation_type == 'tanh':
        network.set_logistic(False)

    # create neurons
    neuron_type = None
    for ng in chromo.node_genes[num_inputs:]:
        if ng.type == 'OUTPUT':
            neuron_type = 1
        else:
            neuron_type = 0
        #print 'Creating neuron: ', ng.id-num_inputs-1, ng.bias, ng.response, neuron_type
        network.setNeuronParameters(ng.id-num_inputs-1, ng.bias, ng.response, neuron_type)

    # create connections
    for cg in chromo.conn_genes:
        if cg.enabled:
            if cg.innodeid-1 < num_inputs:
                # set sensory input
                network.set_sensory_weight(cg.innodeid-1, cg.outnodeid-num_inputs-1, cg.weight)
                #print "Sensory: ", cg.innodeid-1, cg.outnodeid-num_inputs-1, cg.weight
            else:
                # set interneuron connection
                network.SetConnectionWeight(cg.innodeid-num_inputs-1, cg.outnodeid-num_inputs-1, cg.weight)
                #print "Inter..: ", cg.innodeid-num_inputs, cg.outnodeid-num_inputs-1, cg.weight

    return network

if __name__ == "__main__":
    # setting a network manually
    network = ctrnn.CTRNN(0,2)
    network.set_logistic(True)
    network.set_euler(0.05) # integrate using Euler's method
    #network.set_rk4(0.05)

    network.setNeuronParameters(0, -2.75, 1.0, 1)
    network.setNeuronParameters(1, -1.75, 1.0, 1)

    network.set_neuron_state(0, -0.084000643)
    network.set_neuron_state(1, -0.408035109)

    network.SetConnectionWeight(0, 0, 4.5)
    network.SetConnectionWeight(0, 1, -1.0)
    network.SetConnectionWeight(1, 0, 1.0)
    network.SetConnectionWeight(1, 1, 4.5)

    print "%2.17f %2.17f" %(network.NeuronOutput(0), network.NeuronOutput(1))
    for i in range(100000):
        output = network.pactivate([])
        print "%2.17f %2.17f" %(output[0], output[1])
