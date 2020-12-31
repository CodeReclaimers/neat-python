"""

    Hoverboard: Flight Deviance (Double Fitness), using NSGA-II

    Small example tool to control the hoverboard game using NEAT.
    It uses the NSGA2Reproduction method, with two fitness values: flight time and total distance from the center.

    # USAGE:
    > python evolve-flightdvnc.py <ANGLE>
    > python evolve-flightdvnc.py --help

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
CHECKPOINT_FOLDER = 'checkpoint-flightdvnc'
CONFIG_FILE = 'config-nsga2'

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
    # Run the game and calculate fitness (list)
    genome.fitness = neat.nsga2.NSGA2Fitness(0,0)
    last_pos = (0.5,0.5)
    while(True):
        # Activate Neural Network
        output = net.activate([game.hoverboard.velocity[0], game.hoverboard.velocity[1], game.hoverboard.ang_velocity, game.hoverboard.normal[0], game.hoverboard.normal[1]])

        # Update game state from output and then update physics
        game.hoverboard.set_thrust(output[0], output[1])
        game.update(GAME_TIME_STEP)

        # Fitness 0: flight time
        # Fitness 1: movement from last frame
        genome.fitness.add( GAME_TIME_STEP,
                            #-math.sqrt((game.hoverboard.x-last_pos[0])**2+(game.hoverboard.y-last_pos[1])**2) )
                            -math.sqrt((game.hoverboard.x-0.5)**2+(game.hoverboard.y-0.5)**2) ) # distance from center

        last_pos = (game.hoverboard.x, game.hoverboard.y)

        # End of game
        if (game.reset_flag): break

    #genome.fitness.values[1] /= genome.fitness.values[0]

# Evaluate generation
def eval_genomes(genomes, config):
    # Evaluate each genome
    for genome_id, genome in genomes:
        eval(genome, config)
    # NSGA-II required step: non-dominated sorting
    population.reproduction.sort(genomes)

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
    # Here's where we load the NSGA-II reproduction module
    config = neat.Config(neat.DefaultGenome, neat.nsga2.NSGA2Reproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_FILE)

    # Create the population or load from checkpoint
    global population
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
