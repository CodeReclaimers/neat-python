import math
import pickle
import pygame

from neat import nn, population, config

W, H = 128, 128
WW, WH = 1280, 1024
scheme = 'mono'

num_cols = int(math.floor((WW - 16) / (W + 4)))
num_rows = int(math.floor((WH - 16) / (H + 4))) - 1


def make_mono_image(net):
    image = pygame.Surface((W, H))
    for r in range(H):
        y = -2.0 + 4.0 * r / (H - 1)
        for c in range(W):
            x = -2.0 + 4.0 * c / (W - 1)
            output = net.serial_activate([x, y])
            gray = 255 if output[0] > 0.0 else 0
            image.set_at((r, c), (gray, gray, gray))

    return image


def make_gray_image(net):
    image = pygame.Surface((W, H))
    for r in range(H):
        y = -1.0 + 2.0 * r / (H - 1)
        for c in range(W):
            x = -1.0 + 2.0 * c / (W - 1)
            output = net.serial_activate([x, y])
            gray = int(round((output[0] + 1.0) * 255 / 2.0))
            gray = max(0, min(255, gray))
            image.set_at((r, c), (gray, gray, gray))

    return image


def make_color_image(net):
    image = pygame.Surface((W, H))
    for r in range(H):
        y = -1.0 + 2.0 * r / (H - 1)
        for c in range(W):
            x = -1.0 + 2.0 * c / (W - 1)
            output = net.serial_activate([x, y])
            red = int(round((output[0] + 1.0) * 255 / 2.0))
            green = int(round((output[1] + 1.0) * 255 / 2.0))
            blue = int(round((output[2] + 1.0) * 255 / 2.0))
            red = max(0, min(255, red))
            green = max(0, min(255, green))
            blue = max(0, min(255, blue))
            image.set_at((r, c), (red, green, blue))

    return image


if scheme == 'gray':
    make_image = make_gray_image
elif scheme == 'color':
    make_image = make_color_image
elif scheme == 'mono':
    make_image = make_mono_image
else:
    raise Exception("Unexpected scheme: " + repr(scheme))


def eval_fitness(genomes):
    global W, H
    selected = []
    rects = []
    buttons = []
    for n, g in enumerate(genomes):
        net = nn.create_feed_forward_phenotype(g)
        buttons.append(make_image(net))
        selected.append(False)
        row, col = divmod(n, num_cols)
        rects.append(pygame.Rect(4 + (W + 4) * col, 4 + (H + 4) * row, W, H))

    pygame.init()
    screen = pygame.display.set_mode((WW, WH))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                break

            if event.type == pygame.MOUSEBUTTONDOWN:
                print event
                clicked_button = -1
                for n, button in enumerate(buttons):
                    if rects[n].collidepoint(pygame.mouse.get_pos()):
                        clicked_button = n
                        break

                if event.button == 1:
                    selected[clicked_button] = not selected[clicked_button]
                else:
                    net = nn.create_feed_forward_phenotype(genomes[clicked_button])
                    oldW, oldH = W, H
                    W, H = 1000, 1000
                    image = make_image(net)
                    pygame.image.save(image, "rendered-%d.png" % genomes[clicked_button].ID)
                    W, H = oldW, oldH

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
            pygame.image.save(buttons[n], "image-%d.png" % g.ID)
            with open("genome-%d.bin" % g.ID, "wb") as f:
                pickle.dump(g, f, 2)
        else:
            g.fitness = 0.0


def run():
    cfg = config.Config('config')

    if scheme == 'color':
        cfg.output_nodes = 3
    else:
        cfg.output_nodes = 1

    cfg.pop_size = num_cols * num_rows
    pop = population.Population(cfg)
    while 1:
        pop.epoch(eval_fitness, 1)

if __name__ == '__main__':
    run()