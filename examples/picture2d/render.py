import pickle
import pygame

import evolve
from neat import nn, visualize

evolve.W = 1000
evolve.H = 1000

with open("genome-3575.bin", "rb") as f:
    g = pickle.load(f)
    print g
    node_names = {0: 'x', 1: 'y', 2: 'gray'}
    visualize.draw_net(g, view=True, filename="picture-net.gv",
                       show_disabled=False, prune_unused=True, node_names=node_names)

    net = nn.create_feed_forward_phenotype(g)
    image = evolve.make_gray_image(net)
    pygame.image.save(image, "rendered-%d.png" % g.ID)
