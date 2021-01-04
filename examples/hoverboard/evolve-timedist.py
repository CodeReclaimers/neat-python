"""

    Hoverboard: evolve Flight Time + Distance from Center (2 fitnesses), using NSGA-II

    Small example tool to control the hoverboard game using NEAT.
    It uses the NSGA2Reproduction method, with two fitness values: flight time and average squared distance from the center.

    Each genome is evaluated starting from 5 preset points (including center),
    with a given starting angle and it's inverse (ex. 5° and -5°).
    The total fitness is the average for the 10 runs.

    # USAGE:
    > python evolve-timedist.py <ANGLE>
    > python evolve-timedist.py --help

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
CHECKPOINT_FOLDER = 'checkpoint-timedist'
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
    # Genome fitness to be accumulated
    genome.fitness = neat.nsga2.NSGA2Fitness(0,0)
    # Eval starting from 5 points
    for start in [(0.5,0.5),(0.25,0.25),(0.25,0.75),(0.75,0.25),(0.75,0.75)]:
        # Eval with angle and inverse angle
        for angle in [GAME_START_ANGLE,-GAME_START_ANGLE]:
            # Create game
            game = Game(GAME_START_ANGLE,False,start=start)
            # Run the game and calculate fitness (list)
            fitness = [0,0]
            while(True):
                # Activate Neural Network
                dir = [0.5-game.hoverboard.x, 0.5-game.hoverboard.y]
                output = net.activate([game.hoverboard.velocity[0], game.hoverboard.velocity[1], game.hoverboard.ang_velocity, game.hoverboard.normal[0], game.hoverboard.normal[1], dir[0], dir[1]])

                # Update game state from output and then update physics
                game.hoverboard.set_thrust(output[0], output[1])
                game.update(GAME_TIME_STEP)

                # Fitness 0: flight time
                fitness[0] += GAME_TIME_STEP
                # Fitness 1: distance from center (target)
                dist = dir[0]**2+dir[1]**2
                fitness[1] -= dist

                # End of game
                if (game.reset_flag): break
            # Add fitness to genome fitness
            genome.fitness.add(fitness[0],fitness[1])

    # Take average of runs
    genome.fitness.values[0] /= 10
    genome.fitness.values[1] /= 10
    genome.fitness.values[1] /= genome.fitness.values[0]

# Evaluate generation
def eval_genomes(genomes, config):
    # Evaluate each genome
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
        population.add_reporter(GameReporter(population, GAME_TIME_STEP*(10 if FAST_FORWARD else 1), GAME_START_ANGLE, True))

    # Add a checkpointer to save population pickles
    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)
    population.add_reporter(neat.Checkpointer(1,filename_prefix=os.path.join(CHECKPOINT_FOLDER,'gen-')))

    # Run until a solution is found.
    winner = population.run(eval_genomes)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

main()
