"""

    Hoverboard: evolve Flight Time (1 fitness)

    Small example tool to control the hoverboard game using NEAT.
    It uses the DefaultReproduction method, with a single fitness value: flight time.

    Each genome is evaluated starting from the center, with a given starting angle.

    # USAGE:
    > python evolve-time.py <ANGLE>
    > python evolve-time.py --help

    @author: Hugo Aboud (@hugoaboud)

"""

from __future__ import print_function

##  DEBUG
##  Uses local version of neat-python
import sys
sys.path.append('../../')
##  DEBUG
import neat

import os
import math
import argparse

from hoverboard import Game
from visualize import GameReporter, watch

# General Parameters

GAME_TIME_STEP = 0.001
CHECKPOINT_FOLDER = 'checkpoint-time'
CONFIG_FILE = 'config-default'

# CLI Parameters

GAME_START_ANGLE = 0
SILENT = False
FAST_FORWARD = False

##
#   Evaluation
##

# Evaluate genome
def eval(genome, config):
    # Create network
    net = neat.nn.RecurrentNetwork.create(genome, config)
    # Create game
    game = Game(GAME_START_ANGLE,False)
    # Run the game and calculate fitness
    genome.fitness = 0
    while(True):
        # Activate Neural Network
        output = net.activate([game.hoverboard.velocity[0], game.hoverboard.velocity[1], game.hoverboard.ang_velocity, game.hoverboard.normal[0], game.hoverboard.normal[1]])

        # Update game state from output and then update physics
        game.hoverboard.set_thrust(output[0], output[1])
        game.update(GAME_TIME_STEP)

        # Fitness (best option): flight time
        genome.fitness += GAME_TIME_STEP

        # Fitness (alternatives)
        #genome.fitness -= math.sqrt((game.hoverboard.x-0.5)**2+(game.hoverboard.y-0.5)**2)*GAME_TIME_STEP
        #genome.fitness -= (game.hoverboard.normal[0]**2)
        #genome.fitness -= math.sqrt(game.hoverboard.velocity[0]**2+game.hoverboard.velocity[1]**2)
        #genome.fitness -= game.hoverboard.ang_velocity**2

        # End of game
        if (game.reset_flag): break

# Evaluate generation
def eval_genomes(genomes, config):
    # Evaluate each genome looking for the best
    for genome_id, genome in genomes:
        eval(genome, config)

##
#   Main
##

def main():

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Example of evolving a Neural Network using neat-python to play a 2D hoverboard game.')
    parser.add_argument('angle', help="Starting angle of the platform")
    parser.add_argument('-c', '--checkpoint', help="Number of a checkpoint on the 'checkpoint-reference' folder to start from")
    parser.add_argument('-s', '--silent', help="Don't watch the game", nargs='?', const=True, type=bool)
    parser.add_argument('-f', '--fastfwd', help="Fast forward the game preview (10x)", nargs='?', const=True, type=bool)
    args = parser.parse_args()

    # Store global parameters
    global GAME_START_ANGLE
    global SILENT
    global FAST_FORWARD
    GAME_START_ANGLE = float(args.angle)
    SILENT = bool(args.silent)
    FAST_FORWARD = bool(args.fastfwd)

    # Load neat configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_FILE)

    # Create the population or load from checkpoint
    if (args.checkpoint != None):
        population = neat.Checkpointer.restore_checkpoint(os.path.join(CHECKPOINT_FOLDER,'gen-'+str(args.checkpoint)), config)
    else:
        population = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    population.add_reporter(neat.StdOutReporter(False))

    # Add a game reporter to watch the game post evaluation
    if (not SILENT):
        population.add_reporter(GameReporter(population, GAME_TIME_STEP*(10 if FAST_FORWARD else 1), GAME_START_ANGLE))

    # Add a checkpointer to save population pickles
    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)
    population.add_reporter(neat.Checkpointer(1,filename_prefix=os.path.join(CHECKPOINT_FOLDER,'gen-')))

    # Run until a solution is found.
    winner = population.run(eval_genomes)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

main()
