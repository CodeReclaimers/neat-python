import os
import cPickle

import gizeh as gz
import moviepy.editor as mpy

from neat.config import Config
from neat.nn import nn_pure as nn
from neat import visualize
from evolve_single_pole import CartPole

W, H = 300, 100
D = 15  # duration in seconds
SCALE = 300 / 6

cart = gz.rectangle(SCALE * 0.5, SCALE * 0.25, xy=(150, 80), stroke_width=1, fill=(0, 1, 0))
pole = gz.rectangle(SCALE * 0.1, SCALE * 1.0, xy=(150, 55), stroke_width=1, fill=(1, 1, 0))

# random.seed(0)

# load the winner
with open('winner_chromosome') as f:
    c = cPickle.load(f)

# Visualize the best network.
visualize.draw_net(c, view=True)

print 'Loaded chromosome:'
print c

local_dir = os.path.dirname(__file__)
config = Config(os.path.join(local_dir, 'spole_config'))
net = nn.create_phenotype(c)

sim = CartPole()


def make_frame(t):
    inputs = sim.get_scaled_state()
    action = net.pactivate(inputs)
    # action[0] += 0.4 * (np.random.random() - 0.5)

    # Apply action to the simulated cart-pole
    force = 10.0 if action[0] > 0.5 else -10.0
    sim.step(force)

    surface = gz.Surface(W, H, bg_color=(1, 1, 1))

    # Convert position to display units
    visX = SCALE * sim.x

    # Draw cart.
    group = gz.Group((cart,)).translate((visX, 0))
    group.draw(surface)

    # Draw pole.
    group = gz.Group((pole,)).translate((visX, 0)).rotate(sim.theta, center=(150 + visX, 80))
    group.draw(surface)

    return surface.get_npimage()


clip = mpy.VideoClip(make_frame, duration=D)
clip.write_videofile("gizeh.mp4", codec="mpeg4", fps=50)
