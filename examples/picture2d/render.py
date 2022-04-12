import os
import pickle

import evolve_interactive as evolve
import neat

evolve.W = 1000
evolve.H = 1000

with open("genome-4432-586.bin", "rb") as f:
    g = pickle.load(f)
    print(g)
    node_names = {0: 'x', 1: 'y', 2: 'gray'}
    # visualize.draw_net(g, view=True, filename="picture-net.gv",
    #                   show_disabled=False, prune_unused=True, node_names=node_names)

    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'interactive_config_gray')
    # Note that we provide the custom stagnation class to the Config constructor.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, evolve.InteractiveStagnation,
                         config_path)

    net = neat.nn.FeedForwardNetwork.create(g, config)

    # pb = evolve.PictureBreeder(128, 128, 1500, 1500, 1280, 1024, 'color', 4)
    # pb.make_high_resolution(g, config)

    names = {-1: 'x0', -2: 'y0', 0: 'gray'}

    for node, act_func, agg_func, bias, response, links in net.node_evals:
        node_inputs = []
        for i, w in links:
            input_name = names[i] if i in names else 'node%d' % i
            node_inputs.append('%s * %.3f' % (input_name, w))
        s = '%s(%s)' % (agg_func.__name__, ', '.join(node_inputs))
        node_name = names[node] if node in names else 'node%d' % node
        print('%s = %s(%.3f + %.3f * %s)' % (node_name, act_func.__name__, bias, response, s))
