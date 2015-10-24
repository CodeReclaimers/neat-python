import os
from random import randint
import cPickle as pickle

import numpy as np
import gizeh as gz
import moviepy.editor as mpy

from neat import config, chromosome, genome
from neat.nn import nn_pure as nn
from evolve_single_pole import cart_pole, angle_limit

W, H = 300, 100
D = 15  # duration in seconds
SCALE = 300 / 6

cart = gz.rectangle(SCALE * 0.5, SCALE * 0.25, xy=(150, 80), stroke_width=1, fill=(0, 1, 0))
pole = gz.rectangle(SCALE * 0.1, SCALE * 1.0, xy=(150, 55), stroke_width=1, fill=(1, 1, 0))

# random.seed(0)

chromosome.node_gene_type = genome.NodeGene

# load the winner
with open('winner_chromosome') as f:
    c = pickle.load(f)

print 'Loaded chromosome:'
print c

local_dir = os.path.dirname(__file__)
config.load(os.path.join(local_dir, 'spole_config'))
net = nn.create_phenotype(c)

# initial conditions (as used by Stanley)
x = randint(0, 4800) / 1000.0 - 2.4
x_dot = randint(0, 4000) / 1000.0 - 2.0
theta = randint(0, 400) / 1000.0 - 0.2
theta_dot = randint(0, 3000) / 1000.0 - 1.5

last_t = 0


def make_frame(t):
    global x, x_dot, theta, theta_dot, last_t

    # maps into [0,1]
    inputs = [(x + 2.4) / 4.8,
              (x_dot + 0.75) / 1.5,
              (theta + angle_limit) / 0.41,
              (theta_dot + 1.0) / 2.0]

    action = net.pactivate(inputs)
    action[0] += 0.4 * (np.random.random() - 0.5)

    # Apply action to the simulated cart-pole
    x, x_dot, theta, theta_dot = cart_pole(action[0], x, x_dot, theta, theta_dot)

    surface = gz.Surface(W, H, bg_color=(1, 1, 1))

    # Convert position to display units
    visX = SCALE * x

    # Draw cart.
    group = gz.Group((cart,)).translate((visX, 0))
    group.draw(surface)

    # Draw pole.
    group = gz.Group((pole,)).translate((visX, 0)).rotate(theta, center=(150 + visX, 80))
    group.draw(surface)

    return surface.get_npimage()


clip = mpy.VideoClip(make_frame, duration=D)
clip.write_videofile("gizeh.mp4", codec="mpeg4", fps=50)
