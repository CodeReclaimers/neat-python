# This example follows from Beer's C++ source code available at:
# http://mypage.iu.edu/~rdbeer/
import numpy as np
import matplotlib.pyplot as plt

from neat import nn
from neat.ctrnn import CTNeuron

# create two output neurons (they won't receive any external inputs)
N1 = CTNeuron('OUTPUT', 1, -2.75, 1.0, 'exp', 0.5)
N2 = CTNeuron('OUTPUT', 2, -1.75, 1.0, 'exp', 0.5)
N1.set_init_state(-0.084000643)
N2.set_init_state(-0.408035109)

neurons_list = [N1, N2]
# create some synapses
conn_list = [(1, 1, 4.5), (1, 2, -1.0), (2, 1, 1.0), (2, 2, 4.5)]
# create the network
net = nn.Network(neurons_list, conn_list)
# activates the network
print "%.17f %.17f" % (N1.output, N2.output)
outputs = []
for i in xrange(1000):
    output = net.parallel_activate()
    outputs.append(output)
    print "%.17f %.17f" % (output[0], output[1])

outputs = np.array(outputs).T

plt.title("CTRNN model")
plt.ylabel("Outputs")
plt.xlabel("Time")
plt.grid()
plt.plot(outputs[0], "g-", label="output 0")
plt.plot(outputs[1], "r-", label="output 1")
plt.legend(loc="best")
plt.show()
plt.close()
