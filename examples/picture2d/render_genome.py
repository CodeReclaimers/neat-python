import os
import pickle

import pygame

import common
import neat

pygame.init()

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'interactive_config_gray')


class InteractiveStagnation(object):
    """
    This class is used as a drop-in replacement for the default species stagnation scheme.

    A species is only marked as stagnant if the user has not selected one of its output images
    within the last `max_stagnation` generations.
    """

    def __init__(self, config, reporters):
        self.max_stagnation = int(config.get('max_stagnation'))
        self.reporters = reporters

    @classmethod
    def parse_config(cls, param_dict):
        config = {'max_stagnation': 15}
        config.update(param_dict)

        return config

    @classmethod
    def write_config(cls, f, config):
        f.write('max_stagnation       = {}\n'.format(config['max_stagnation']))

    def update(self, species_set, generation):
        result = []
        for s in species_set.species.values():
            # If any member of the species is selected (i.e., has a fitness above zero),
            # mark the species as improved.
            for m in s.members.values():
                if m.fitness > 0:
                    s.last_improved = generation
                    break

            stagnant_time = generation - s.last_improved
            is_stagnant = stagnant_time >= self.max_stagnation
            result.append((s.key, s, is_stagnant))

        return result


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, InteractiveStagnation,
                     config_path)

fn = 'rendered/genome-26636-1634.bin'
genome = pickle.load(open(fn, 'rb'))

m = 2048
image_data = common.eval_gray_image(genome, config, m, m)

image = pygame.Surface((m, m), depth=8)
palette = tuple([(i, i, i) for i in range(256)])
image.set_palette(palette)

for r, row in enumerate(image_data):
    for c, color in enumerate(row):
        image.set_at((r, c), color)

pygame.image.save(image, fn + "highres.png")

print(genome)
