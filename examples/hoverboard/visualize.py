"""

    neat-python hoverboard visualize tool

    Small tool for watching neat-python genomes play the hoverboard game.
    It takes the genomes from checkpoints, it also plots the fitness info.

    # USAGE:
    > python visualize.py <EXPERIMENT> <ANGLE>
    > python visualize.py --help

    @author: Hugo Aboud (@hugoaboud)

"""

import os
import argparse
import neat
import math

from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from hoverboard import Game
from gui import NeuralNetworkGUI

# General Parameters

GAME_TIME_STEP = 0.001

# CLI Parameters

GAME_START_ANGLE = 0
FAST_FORWARD = False
EXPERIMENT = 'flightime'

##
#   Data
#   parse populations from checkpoints
##

def load_checkpoints(folder):
    print("Loading checkpoints from {0}...".format(folder))
    # load generations from file
    checkpoints = []
    files = os.listdir(folder)
    # progress bar vars
    step = len(files)/46
    t = 0
    print('[', end='', flush=True)
    for filename in files:
        # load checkpoint and append to list
        checkpoint = neat.Checkpointer.restore_checkpoint(os.path.join(folder,filename))
        checkpoints.append(checkpoint)
        # update progress bar
        t += 1
        if (t > step):
            t -= step
            print('.', end='', flush=True)
    print(']')
    # Sort checkpoints by generation id
    checkpoints.sort(key = lambda g: g.generation)
    return checkpoints

##
#   Plot
#   scientific plot of fitness using pyplot
##

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

##
#   Watch
#   watch a genome play the game
##
def watch(config, time_step, generation, species, genome, start_angle):
    # create a recurrent network
    net = neat.nn.RecurrentNetwork.create(genome, config)
    # create a network GUI to render the topology and info
    ui = NeuralNetworkGUI(generation, genome, species, net)
    # create a Game with frontend enabled, and the GUI above
    game = Game(start_angle,True,ui)
    # run the game until reset
    while(True):
        # activate network
        output = net.activate([game.hoverboard.velocity[0], game.hoverboard.velocity[1], game.hoverboard.ang_velocity, game.hoverboard.normal[0], game.hoverboard.normal[1]])
        # output to hoverboard thrust
        game.hoverboard.set_thrust(output[0], output[1])
        # update game manually from time step
        game.update(time_step)
        # if game reseted, break
        if (game.reset_flag): break

##
#   Main
##

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Tool for visualizing the neat-python checkpoints playing the hoverboard game.')
    parser.add_argument('angle', help="Starting angle of the platform")
    parser.add_argument('experiment', help="Experiment prefix: (flightime,rundvnc), default: flighttime", const='flighttime', nargs='?')
    parser.add_argument('-f', '--fastfwd', help="Fast forward the game preview (2x)", nargs='?', const=True, type=bool)
    args = parser.parse_args()

    # Store global parameters
    global GAME_START_ANGLE
    global FAST_FORWARD
    GAME_START_ANGLE = float(args.angle)
    FAST_FORWARD = bool(args.fastfwd)

    # Check experiment argument
    global EXPERIMENT
    if (args.experiment is not None):
        EXPERIMENT = str(args.experiment)
        if (EXPERIMENT != 'flightime'):
            print("ERROR: Invalid experiment '" + EXPERIMENT + "'")
            return

    # load data
    checkpoints = load_checkpoints('checkpoint-'+EXPERIMENT)

    # create neat config from file
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-'+EXPERIMENT)

    # run game for the best genome of each checkpoint
    # if it's not the same as the last one
    last_genome = None
    for checkpoint in checkpoints:
        # skip repeated genomes
        if (checkpoint.best_genome.key != last_genome):
            last_genome = checkpoint.best_genome.key
        else:
            continue
        # get species id
        species = checkpoint.species.get_species_id(checkpoint.best_genome.key)
        # watch the genome play
        watch(config, GAME_TIME_STEP, checkpoint.generation, species, checkpoint.best_genome, GAME_START_ANGLE)

    # scientific plot
    plot(checkpoints, 'reference')

if __name__ == "__main__":
   main()
