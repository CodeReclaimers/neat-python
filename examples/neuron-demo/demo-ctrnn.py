from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import neat
from neat.activations import sigmoid_activation

# Create a fully-connected network of two neurons with no external inputs.
node1_inputs = [(1, 0.9), (2, 0.2)]
node2_inputs = [(1, -0.2), (2, 0.9)]

node_evals = {1: neat.ctrnn.CTRNNNodeEval(0.01, sigmoid_activation, sum, -2.75 / 5.0, 1.0, node1_inputs),
              2: neat.ctrnn.CTRNNNodeEval(0.01, sigmoid_activation, sum, -1.75 / 5.0, 1.0, node2_inputs)}

net = neat.ctrnn.CTRNN([], [1, 2], node_evals)

init1 = 0.0
init2 = 0.0

net.set_node_value(1, init1)
net.set_node_value(2, init2)

times = [0.0]
outputs = [[init1, init2]]
for i in range(1250):
    output = net.advance([], 0.002, 0.002)
    times.append(net.time_seconds)
    outputs.append(output)
    print("{0:.7f} {1:.7f}".format(output[0], output[1]))

outputs = np.array(outputs).T

plt.title("CTRNN model")
plt.ylabel("Outputs")
plt.xlabel("Time")
plt.grid()
plt.plot(times, outputs[0], "g-", label="output 0")
plt.plot(times, outputs[1], "r-", label="output 1")
plt.legend(loc="best")

plt.figure()
plt.ylabel("Output 0")
plt.xlabel("Output 1")
plt.grid()
plt.plot(outputs[0], outputs[1], "g-")

plt.show()
plt.close()
