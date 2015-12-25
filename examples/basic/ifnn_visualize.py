from __future__ import print_function

import matplotlib.pyplot as plt

from neat.ifnn import IFNeuron

n = IFNeuron()
times = []
currents = []
potentials = []
fired = []
for i in range(1000):
    times.append(1.0 * i)

    n.current = 0.0 if i < 100 or i > 800 else 16.0
    currents.append(n.current)

    n.advance()

    potentials.append(n.potential)
    fired.append(1.0 if n.has_fired else 0.0)

plt.subplot(3, 1, 1)
plt.title("IFNN model")
plt.ylabel("Input current")
plt.ylim(0, 20)
plt.grid()
plt.plot(times, currents, "b-", label="current")

plt.subplot(3, 1, 2)
plt.ylabel("Potential")
plt.grid()
plt.plot(times, potentials, "g-", label="potential")

plt.subplot(3, 1, 3)
plt.ylabel("Fired")
plt.xlabel("Time (msec)")
plt.grid()
plt.plot(times, fired, "r-", label="fired")

plt.show()
plt.close()
