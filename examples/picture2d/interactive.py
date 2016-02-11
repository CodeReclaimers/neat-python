"""
This is an example that amounts to an offline picbreeder.org without any nice features. :)

Left-click on thumbnails to pick images to breed for next generation, right-click to
render a high-resolution version of an image.  Genomes and images chosen for breeding
and rendering are saved to disk.
"""
import math
import os
import pickle
import pygame

# TODO: Allow user-defined species stagnation test.  It's annoying to have an interesting species
# die out in this application just because the fitness can't go up forever.

from multiprocessing import Pool
from neat import population, config
from common import eval_mono_image, eval_gray_image, eval_color_image


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

    def make_thumbnails(self, genomes):
        img_func = eval_mono_image
        if self.scheme == 'gray':
            img_func = eval_gray_image
        elif self.scheme == 'color':
            img_func = eval_color_image

        jobs = []
        for g in genomes:
            jobs.append(self.pool.apply_async(img_func, (g, self.thumb_width, self.thumb_height)))

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

    def make_high_resolution(self, genome):
        # Make sure the output directory exists.
        if not os.path.isdir('rendered'):
            os.mkdir('rendered')

        if self.scheme == 'gray':
            image_data = eval_gray_image(genome, self.full_width, self.full_height)
        elif self.scheme == 'color':
            image_data = eval_color_image(genome, self.full_width, self.full_height)
        else:
            image_data = eval_mono_image(genome, self.full_width, self.full_height)

        image = self.make_image_from_data(image_data)
        pygame.image.save(image, "rendered/rendered-{}-{}.png".format(os.getpid(), genome.ID))

        with open("rendered/genome-{}-{}.bin".format(os.getpid(), genome.ID), "wb") as f:
            pickle.dump(genome, f, 2)

    def eval_fitness(self, genomes):
        selected = []
        rects = []
        for n, g in enumerate(genomes):
            selected.append(False)
            row, col = divmod(n, self.num_cols)
            rects.append(pygame.Rect(4 + (self.thumb_width + 4) * col,
                                     4 + (self.thumb_height + 4) * row,
                                     self.thumb_width, self.thumb_height))

        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))

        buttons = self.make_thumbnails(genomes)

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
                        self.make_high_resolution(genomes[clicked_button])

            if running:
                screen.fill((128,128,192))
                for n, button in enumerate(buttons):
                    screen.blit(button, rects[n])
                    if selected[n]:
                        pygame.draw.rect(screen, (255, 0, 0), rects[n], 3)
                pygame.display.flip()

        for n, g in enumerate(genomes):
            if selected[n]:
                g.fitness = 1.0
                pygame.image.save(buttons[n], "image-{}.{}.png".format(os.getpid(), g.ID))
                with open("genome-{}-{}.bin".format(os.getpid(), g.ID), "wb") as f:
                    pickle.dump(g, f, 2)
            else:
                g.fitness = 0.0

def run():
    # 128x128 thumbnails, 1500x1500 rendered images, 1100x810 viewer, grayscale images.
    pb = PictureBreeder(128, 128, 1500, 1500, 1100, 810, 'gray', 4)
    cfg = config.Config('config')

    # Make sure the network has the expected number of outputs.
    if pb.scheme == 'color':
        cfg.output_nodes = 3
    else:
        cfg.output_nodes = 1

    cfg.pop_size = pb.num_cols * pb.num_rows
    pop = population.Population(cfg)
    while 1:
        pop.run(pb.eval_fitness, 1)

if __name__ == '__main__':
    run()