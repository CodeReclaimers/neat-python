from __future__ import print_function

from neat.ctrnn import CTNeuron, Network


def test_basic():
    # create two output neurons (they won't receive any external inputs)
    n1 = CTNeuron('OUTPUT', 1, -2.75, 1.0, 'exp', 0.5)
    n2 = CTNeuron('OUTPUT', 2, -1.75, 1.0, 'exp', 0.5)
    n1.set_init_state(-0.084000643)
    n2.set_init_state(-0.408035109)

    neurons_list = [n1, n2]
    # create some synapses
    conn_list = [(1, 1, 4.5), (1, 2, -1.0), (2, 1, 1.0), (2, 2, 4.5)]
    # create the network
    net = Network(neurons_list, conn_list)
    # activates the network
    print("%.17f %.17f" % (n1.output, n2.output))
    outputs = []
    for i in range(1000):
        output = net.parallel_activate()
        outputs.append(output)
        print("%.17f %.17f" % (output[0], output[1]))
