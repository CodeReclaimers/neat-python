from neat import visualize
from neat.iznn import Neuron

n = Neuron(10)
spike_train = []
for i in range(1000):
    spike_train.append(n.potential)
    print '%d\t%f' % (i, n.potential)
    n.advance()

visualize.plot_spikes(spike_train, view=True, filename='spiking_neuron.svg')
