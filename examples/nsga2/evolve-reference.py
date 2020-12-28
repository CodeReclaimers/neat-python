"""
    Hoverboard (Single Fitness) Example

    Control the hoverboard game using the DefaultReproduction method,
    with a single fitness value: flight time.

    This is a reference example, without NSGA-II.
"""

from __future__ import print_function
import neat
import os
import math
import argparse

from hoverboard import Game
from visualize import watch

GAME_START_ANGLE = 0
GAME_TIME_STEP = 0.001

SILENT = False
FAST_FORWARD = False

# Eval

BEST = None
GEN = 0
POPULATION = None

# evaluate the genome using a recurrent network
def eval(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    game = Game(GAME_START_ANGLE,False)
    genome.fitness = 0
    while(True):
        output = net.activate([game.hoverboard.velocity[0], game.hoverboard.velocity[1], game.hoverboard.ang_velocity, game.hoverboard.normal[0], game.hoverboard.normal[1]])
        game.hoverboard.set_thrust(output[0], output[1])
        game.update(GAME_TIME_STEP)

        # fitness option 1: flight time
        genome.fitness += GAME_TIME_STEP

        # fitness option 2: deviance from the center
        #genome.fitness -= math.sqrt(game.hoverboard.velocity[0]**2+game.hoverboard.velocity[1]**2)
        #genome.fitness -= math.sqrt((game.hoverboard.x-0.5)**2+(game.hoverboard.y-0.5)**2)*GAME_TIME_STEP
        #genome.fitness -= game.hoverboard.ang_velocity**2
        #genome.fitness -= (game.hoverboard.normal[0]**2)

        if (game.reset_flag): break

# evaluate all genomes in generation
# if the best genome has changed, watch it play
def eval_genomes(genomes, config):
    global BEST
    global GEN
    global POPULATION
    max = None
    for genome_id, genome in genomes:
        eval(genome, config)
        if (max == None or genome.fitness > max.fitness):
            max = genome
    if (not SILENT and max != BEST):
        BEST = max
        species = POPULATION.species.get_species_id(BEST.key)
        watch(GEN, BEST, species, config, GAME_START_ANGLE, GAME_TIME_STEP*(10 if FAST_FORWARD else 1))
    GEN += 1

def main():

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Example of evolving a Neural Network using neat-python to control a 2D hoverboard simulation.')
    parser.add_argument('angle', help="Starting angle of the platform")
    parser.add_argument('-c', '--checkpoint', help="Number of a checkpoint on the 'checkpoint-reference' folder to start from")
    parser.add_argument('-s', '--silent', help="Don't watch the game", nargs='?', const=True, type=bool)
    parser.add_argument('-f', '--fastfwd', help="Fast forward the game preview (10x)", nargs='?', const=True, type=bool)
    args = parser.parse_args()

    global GAME_START_ANGLE
    GAME_START_ANGLE = float(args.angle)

    global SILENT
    SILENT = bool(args.silent)

    global FAST_FORWARD
    FAST_FORWARD = bool(args.fastfwd)

    # Load neat configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-reference')

    # Create the population or load from checkpoint
    global POPULATION
    if (args.checkpoint != None):
        POPULATION = neat.Checkpointer.restore_checkpoint(os.path.join('checkpoint-reference','gen-'+str(args.checkpoint)), config)
    else:
        POPULATION = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    POPULATION.add_reporter(neat.StdOutReporter(False))

    # Add a checkpointer to save population pickles
    if not os.path.exists('checkpoint-reference'):
        os.makedirs('checkpoint-reference')
    POPULATION.add_reporter(neat.Checkpointer(1,filename_prefix='checkpoint-reference/gen-'))

    # Run until a solution is found.
    winner = POPULATION.run(eval_genomes)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

main()
