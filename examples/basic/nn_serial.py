# Example
# from neat import visualize
from neat.nn.nn_pure import FeedForward

nn = FeedForward([2, 10, 3], use_bias=False, activation_type='exp')
##visualize.draw_ff(nn)
print 'Serial activation method: '
for t in range(3):
    print nn.sactivate([1, 1])

    # print 'Parallel activation method: '
    # for t in range(3):
    # print nn.pactivate([1,1])

    # defining a neural network manually
    # neurons = [Neuron('INPUT', 1), Neuron('HIDDEN', 2), Neuron('OUTPUT', 3)]
    # connections = [(1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.5)]

    # net = Network(neurons, connections) # constructs the neural network
    # visualize.draw_ff(net)
    # print net.pactivate([0.04]) # parallel activation method
    # print net # print how many neurons and synapses our network has
