"""
This is an example that amounts to an offline picbreeder.org without any nice features. :)

Left-click on thumbnails to pick images to breed for next generation, right-click to
render a high-resolution version of an image.  Genomes and images chosen for breeding
and rendering are saved to disk.

This example also demonstrates how to customize species stagnation.
"""
import math
import os
import pickle
import pygame

from multiprocessing import Pool
import neat
from neat.six_util import itervalues

from common import eval_mono_image, eval_gray_image, eval_color_image


class InteractiveStagnation(object):
    """
    This class is used as a drop-in replacement for the default species stagnation scheme.

    A species is only marked as stagnant if the user has not selected one of its output images
    within the last `max_stagnation` generations.
    """
    def __init__(self, config, reporters):
        self.max_stagnation = int(config.get('max_stagnation'))

        self.reporters = reporters
        self.stagnant_counts = {}

    @classmethod
    def parse_config(cls, param_dict):
        config = {'max_stagnation': 15}
        config.update(param_dict)

        return config

    @classmethod
    def write_config(cls, f, config):
        max_stagnation = config.get('max_stagnation', 15)
        f.write('max_stagnation       = {}\n'.format(max_stagnation))

    def remove(self, sid):
        self.stagnant_counts.pop(sid, None)

    def update(self, species_set, generation):
        result = []
        for s in itervalues(species_set.species):
            # If any member of the species is selected (i.e., has a fitness above zero), then we reset
            # the stagnation count.  Otherwise we increment the count.
            scount = self.stagnant_counts.get(s.key, 0) + 1
            for m in s.members.values():
                if m.fitness > 0:
                    scount = 0
                    break

            self.stagnant_counts[s.key] = scount

            is_stagnant = scount >= self.max_stagnation
            result.append((s.key, s, is_stagnant))

            if is_stagnant:
                self.remove(s)

        self.reporters.info('Species no improv: {0!r}'.format(self.stagnant_counts))

        return result


class PictureBreeder(object):
    def __init__(self, thumb_width, thumb_height, full_width, full_height,
                 window_width, window_height, scheme, num_workers):
        """
        :param thumb_width: Width of preview image
        :param thumb_height: Height of preview image
        :param full_width: Width of full rendered image
        :param full_height: Height of full rendered image
        :param window_width: Width of the view window
        :param window_height: Height of the view window
        :param scheme: Image type to generate: mono, gray, or color
        """
        self.generation = 0
        self.thumb_width = thumb_width
        self.thumb_height = thumb_height
        self.full_width = full_width
        self.full_height = full_height

        self.window_width = window_width
        self.window_height = window_height

        assert scheme in ('mono', 'gray', 'color')
        self.scheme = scheme

        # Compute the number of thumbnails we can show in the viewer window, while
        # leaving one row to handle minor variations in the population size.
        self.num_cols = int(math.floor((window_width - 16) / (thumb_width + 4)))
        self.num_rows = int(math.floor((window_height - 16) / (thumb_height + 4)))

        self.pool = Pool(num_workers)

    def make_image_from_data(self, image_data):
        # For mono and grayscale, we need a palette because the evaluation function
        # only returns a single integer instead of an (R, G, B) tuple.
        if self.scheme == 'color':
            image = pygame.Surface((self.thumb_width, self.thumb_height))
        else:
            image = pygame.Surface((self.thumb_width, self.thumb_height), depth=8)
            palette = tuple([(i, i, i) for i in range(256)])
            image.set_palette(palette)

        for r, row in enumerate(image_data):
            for c, color in enumerate(row):
                image.set_at((r, c), color)

        return image

    def make_thumbnails(self, genomes, config):
        img_func = eval_mono_image
        if self.scheme == 'gray':
            img_func = eval_gray_image
        elif self.scheme == 'color':
            img_func = eval_color_image

        jobs = []
        for genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(img_func, (genome, config, self.thumb_width, self.thumb_height)))

        thumbnails = []
        for j in jobs:
            # TODO: This code currently generates the image data using the multiprocessing
            # pool, and then does the actual image construction here because pygame complained
            # about not being initialized if the pool workers tried to construct an image.
            # Presumably there is some way to fix this, but for now this seems fast enough
            # for the purposes of a demo.
            image_data = j.get()

            thumbnails.append(self.make_image_from_data(image_data))

        return thumbnails

    def make_high_resolution(self, genome, config):
        genome_id, genome = genome

        # Make sure the output directory exists.
        if not os.path.isdir('rendered'):
            os.mkdir('rendered')

        if self.scheme == 'gray':
            image_data = eval_gray_image(genome, config, self.full_width, self.full_height)
        elif self.scheme == 'color':
            image_data = eval_color_image(genome, config, self.full_width, self.full_height)
        else:
            image_data = eval_mono_image(genome, config, self.full_width, self.full_height)

        image = self.make_image_from_data(image_data)
        pygame.image.save(image, "rendered/rendered-{}-{}.png".format(os.getpid(), genome_id))

        with open("rendered/genome-{}-{}.bin".format(os.getpid(), genome_id), "wb") as f:
            pickle.dump(genome, f, 2)

    def eval_fitness(self, genomes, config):
        selected = []
        rects = []
        for n, (genome_id, genome) in enumerate(genomes):
            selected.append(False)
            row, col = divmod(n, self.num_cols)
            rects.append(pygame.Rect(4 + (self.thumb_width + 4) * col,
                                     4 + (self.thumb_height + 4) * row,
                                     self.thumb_width, self.thumb_height))

        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Interactive NEAT-python generation {0}".format(self.generation))

        buttons = self.make_thumbnails(genomes, config)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    running = False
                    break

                if event.type == pygame.MOUSEBUTTONDOWN:
                    clicked_button = -1
                    for n, button in enumerate(buttons):
                        if rects[n].collidepoint(pygame.mouse.get_pos()):
                            clicked_button = n
                            break

                    if event.button == 1:
                        selected[clicked_button] = not selected[clicked_button]
                    else:
                        self.make_high_resolution(genomes[clicked_button], config)

            if running:
                screen.fill((128, 128, 192))
                for n, button in enumerate(buttons):
                    screen.blit(button, rects[n])
                    if selected[n]:
                        pygame.draw.rect(screen, (255, 0, 0), rects[n], 3)
                pygame.display.flip()

        for n, (genome_id, genome) in enumerate(genomes):
            if selected[n]:
                genome.fitness = 1.0
                pygame.image.save(buttons[n], "image-{}.{}.png".format(os.getpid(), genome_id))
                with open("genome-{}-{}.bin".format(os.getpid(), genome_id), "wb") as f:
                    pickle.dump(genome, f, 2)
            else:
                genome.fitness = 0.0


def run():
    # 128x128 thumbnails, 1500x1500 rendered images, 1100x810 viewer, grayscale images, 4 worker processes.
    pb = PictureBreeder(128, 128, 1500, 1500, 1100, 810, 'gray', 4)

    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'interactive_config')
    # Note that we provide the custom stagnation class to the Config constructor.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, InteractiveStagnation,
                         config_path)

    # Make sure the network has the expected number of outputs.
    if pb.scheme == 'color':
        config.output_nodes = 3
    else:
        config.output_nodes = 1

    config.pop_size = pb.num_cols * pb.num_rows
    pop = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    while 1:
        pb.generation = pop.generation + 1
        pop.run(pb.eval_fitness, 1)

if __name__ == '__main__':
    run()
