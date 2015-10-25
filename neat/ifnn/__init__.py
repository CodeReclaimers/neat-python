from ifnn_pure import *
from neat.iznn.network import Network


def create_phenotype(chromosome):
    """ Receives a chromosome and returns its phenotype (a neural network) """

    neurons = {}
    input_neurons = []
    output_neurons = []
    for ng in chromosome.node_genes:
        neurons[ng.ID] = Neuron(ng.bias)
        if ng.type == 'INPUT':
            input_neurons.append(neurons[ng.ID])
        elif ng.type == 'OUTPUT':
            output_neurons.append(neurons[ng.ID])

    synapses = [Synapse(neurons[cg.innodeid], neurons[cg.outnodeid], cg.weight) \
                for cg in chromosome.conn_genes if cg.enabled]

    return Network(neurons, input_neurons, output_neurons, synapses)
