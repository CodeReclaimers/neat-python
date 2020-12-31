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

##  DEBUG
##  Uses local version of neat-python
import sys
sys.path.append('../../')
##  DEBUG
import neat
from neat.math_util import mean

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
JUST_PLOT = False
EXPERIMENT = 'flightime'

##
#   Reporter
#   Used to watch the game after each evaluation
##

class GameReporter(neat.reporting.BaseReporter):
    def __init__(self, population, step, angle):
        self.population = population
        self.step = step
        self.angle = angle
        self.best = None
        self.gen = 0
    def post_evaluate(self, config, population, species, best_genome):
        # If best genome has changed, watch it
        if (not self.best or best_genome != self.best):
            self.best = best_genome
            species = self.population.species.get_species_id(self.best.key)
            watch(config, self.step, self.gen, species, self.best, self.angle)
        self.gen += 1

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
    avgs = [mean([f.fitness for _, f in c.population.items()]) for c in checkpoints]

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
    parser.add_argument('-p', '--just_plot', help="Don't watch the game, just plot", nargs='?', const=True, type=bool)
    args = parser.parse_args()

    # Store global parameters
    global GAME_START_ANGLE
    global FAST_FORWARD
    global JUST_PLOT
    GAME_START_ANGLE = float(args.angle)
    FAST_FORWARD = bool(args.fastfwd)
    JUST_PLOT = bool(args.just_plot)

    # Check experiment argument
    global EXPERIMENT
    if (args.experiment is not None):
        EXPERIMENT = str(args.experiment)
        if (EXPERIMENT not in ('flightime','flightdvnc')):
            print("ERROR: Invalid experiment '" + EXPERIMENT + "'")
            return

    # load data
    checkpoints = load_checkpoints('checkpoint-'+EXPERIMENT)

    # create neat config from file
    cfg_file = {'flightime':'config-default',
                'flightdvnc':'config-nsga2'}[EXPERIMENT]
    repro = {'flightime':neat.DefaultReproduction,
             'flightdvnc':neat.nsga2.NSGA2Reproduction}[EXPERIMENT]
    config = neat.Config(neat.DefaultGenome, repro, neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_file)

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
        if (not JUST_PLOT):
            watch(config, GAME_TIME_STEP*(2 if FAST_FORWARD else 1), checkpoint.generation, species, checkpoint.best_genome, GAME_START_ANGLE)

    # scientific plot
    plot(checkpoints, 'reference')

if __name__ == "__main__":
   main()
