from __future__ import print_function

import os
import pickle

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from neat import visualize

with open('spiking-winner.dat', 'rb') as f:
    winner, simulated = pickle.load(f)


def voltage_to_color(voltage):
    if voltage >= -30.0:
        r, g, b = 0, 255, 0
    else:
        scaled_voltage = (voltage + 70.0) / 40.0
        r = 255
        g = max(0, min(255, int(255 * (1 - scaled_voltage))))
        b = g

    return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)


node_names = {0: 'A', 1: 'B', 2: 'Out1', 3: 'Out2'}
filenames = []
for r, (inputData, outputData, t0, t1, v0, v1, neuron_data) in enumerate(simulated):
    times = [t for t, v in neuron_data.values()[0]]
    nodes = list(neuron_data.keys())
    time_data = []
    for i, t in enumerate(times):
        tdata = {}
        for n in nodes:
            tdata[n] = neuron_data[n][i][1]

        tdata[0] = 0.0 if inputData[0] == 0 else -75.0
        tdata[1] = 0.0 if inputData[1] == 0 else -75.0

        time_data.append(tdata)

    for ti, (t, tdata) in enumerate(zip(times, time_data)):
        node_colors = {}
        for n, v in tdata.items():
            node_colors[n] = voltage_to_color(v)

        fn = 'spiking-{0:04d}'.format(len(filenames))
        dot = visualize.draw_net(winner, filename=fn, view=False, node_names=node_names, node_colors=node_colors,
                                 fmt='png', show_disabled=False, prune_unused=True)
        filenames.append(dot.filename + '.png')

clip = ImageSequenceClip(filenames, fps=30)
clip.write_videofile('spiking.mp4', codec="mpeg4", bitrate="2000k")

for fn in filenames:
    os.unlink(fn)
    os.unlink(fn[:-4])
