"""
    Hoverboard (Single Fitness) Example

    Control the hoverboard game using the DefaultReproduction method,
    with a single fitness value: flight time.

    This is a reference example, without NSGA-II.
"""

from __future__ import print_function
import neat
import os

from hoverboard import Game

GAME_START_ANGLE = 0
GAME_TIME_STEP = 0.001

# Watch a genome play the game
def watch(genome, config):
    net = neat.ctrnn.CTRNN.create(genome, config, 1)
    game = Game(GAME_START_ANGLE,True)
    while(True):
        output = net.advance([game.hoverboard.velocity[0], game.hoverboard.velocity[1], game.hoverboard.ang_velocity, game.hoverboard.normal[0], game.hoverboard.normal[1]], GAME_TIME_STEP, GAME_TIME_STEP)
        game.hoverboard.set_thrust(output[0], output[1])
        game.update(GAME_TIME_STEP)
        if (game.reset_flag): break

# Eval

BEST = None

def eval(genome, config):
    net = neat.ctrnn.CTRNN.create(genome, config, 1)
    game = Game(GAME_START_ANGLE,False)
    genome.fitness = 0
    while(True):
        output = net.advance([game.hoverboard.velocity[0], game.hoverboard.velocity[1], game.hoverboard.ang_velocity, game.hoverboard.normal[0], game.hoverboard.normal[1]], GAME_TIME_STEP, GAME_TIME_STEP)
        game.hoverboard.set_thrust(output[0], output[1])
        game.update(GAME_TIME_STEP)
        genome.fitness += GAME_TIME_STEP
        if (game.reset_flag): break

def eval_genomes(genomes, config):
    global BEST
    max = None
    for genome_id, genome in genomes:
        eval(genome, config)
        if (max == None or genome.fitness > max.fitness):
            max = genome
    if (max != BEST):
        BEST = max
        watch(max, config)

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Add a checkpointer to save population pickles
if not os.path.exists('checkpoint-reference'):
    os.makedirs('checkpoint-reference')
p.add_reporter(neat.Checkpointer(1,filename_prefix='checkpoint-reference/gen-'))

# Run until a solution is found.
winner = p.run(eval_genomes)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Watch the winner
watch(winner)
