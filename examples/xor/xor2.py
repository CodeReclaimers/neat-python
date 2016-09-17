"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function

import neat

# Inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

# Create a SequentialStatic instance and use it to evolve a network.
n = neat.SequentialStatic(xor_inputs, xor_outputs)
winner = n.evolve(300)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
print('\nOutput:')
for inputs, expected, outputs in n.evaluate(winner):
    print("input {!r}, expected output {!r}, got {!r}".format(inputs, expected, outputs[0]))

print("Total number of evaluations: {}".format(n.total_evaluations))
