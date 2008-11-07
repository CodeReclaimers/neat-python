import ann #importing C++ module

net = ann.ANN(3,3)

# bias input
net.set_sensory_weight(0, 0, 1.5)
net.set_sensory_weight(0, 1, 1.5)
net.set_sensory_weight(0, 2, 1.5)
# input 1
net.set_sensory_weight(1, 0, 1.5)
net.set_sensory_weight(1, 1, 1.5)
# input 2
net.set_sensory_weight(2, 0, 1.5)
net.set_sensory_weight(2, 1, 1.5)
# inter-neurons
net.set_synapse(0, 2, 0.5)
net.set_synapse(1, 2, 0.5)
net.set_synapse(2, 1, -0.5)

# neuron's properties: id, bias, response, type
net.set_neuron(0, 0, 1, 0) # hidden
net.set_neuron(1, 0, 1, 0) # hidden
net.set_neuron(2, 0, 1, 1) # output

for i in range(10):
    print net.sactivate([1.2, 0.2, 0.2])

#print net.get_neuron_output(0)
#print net.get_neuron_output(1)
print net.get_neuron_output(2)
