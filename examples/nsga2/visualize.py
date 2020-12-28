import os
import neat
import math
import pygame
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from hoverboard import Game
from gui import NeuralNetworkGUI

# Data parsing from checkpoints

def load_checkpoints(folder):
    print("Loading checkpoints from {0}...".format(folder), end='', flush=True)
    # load generations from file
    checkpoints = []
    for filename in os.listdir(folder):
        checkpoint = neat.Checkpointer.restore_checkpoint(os.path.join(folder,filename))
        checkpoints.append(checkpoint)
    # sort by generation id
    checkpoints.sort(key = lambda g: g.generation)
    print("OK")
    return checkpoints

# Scientific plot with pyplot

def plot(checkpoints, name):
    ids = [c.generation for c in checkpoints]
    bests = [c.best_genome.fitness for c in checkpoints]
    avgs = [sum([f.fitness for _, f in c.population.items()])/len(c.population) for c in checkpoints]

    fig, ax = plt.subplots(figsize = (10,5))
    ax.set_title("Fitness over Generations")
    ax.plot(ids, bests, color='blue', linewidth=1, label="Best")
    ax.plot(ids, avgs, color='black', linewidth=1, label="Average")
    ax.legend()

    plt.tight_layout()
    fig.savefig(name+'.png', format='png', dpi=300)
    plt.show()
    plt.close()

# Game

GAME_START_ANGLE = 0
GAME_TIME_STEP = 0.001

# Watch a genome play the game
def watch(generation, genome, species, config, start_angle, time_step):
    #net = neat.ctrnn.CTRNN.create(genome, config, 1)
    net = neat.nn.RecurrentNetwork.create(genome, config)
    ui = NeuralNetworkGUI(generation, genome, species, net)
    game = Game(start_angle,True,ui)
    while(True):
        #output = net.advance([game.hoverboard.velocity[0], game.hoverboard.velocity[1], game.hoverboard.ang_velocity, game.hoverboard.normal[0], game.hoverboard.normal[1]], time_step, time_step)
        output = net.activate([game.hoverboard.velocity[0], game.hoverboard.velocity[1], game.hoverboard.ang_velocity, game.hoverboard.normal[0], game.hoverboard.normal[1]])
        game.hoverboard.set_thrust(output[0], output[1])
        game.update(time_step)
        if (game.reset_flag): break

# Main

def main():
    # load data
    checkpoints = load_checkpoints('checkpoint-reference')

    # watch the game
    pygame.init()
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config')
    last_genome = None
    for checkpoint in checkpoints[10:]:
        # skip repeated genomes
        if (checkpoint.best_genome.key != last_genome):
            last_genome = checkpoint.best_genome.key
        else:
            continue
        # get species id
        species = checkpoint.species.get_species_id(checkpoint.best_genome.key)
        # watch the genome play
        watch(checkpoint.generation, checkpoint.best_genome, species, config, GAME_START_ANGLE, GAME_TIME_STEP)

    # scientific plot
    plot(checkpoints, 'reference')

if __name__ == "__main__":
   main()
