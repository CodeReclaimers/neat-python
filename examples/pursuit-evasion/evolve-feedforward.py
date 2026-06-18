"""
2D pursuit-evasion experiment with a feed-forward neural network.

NEAT evolves a pursuer controller that learns to capture an evader
in a bounded 2D continuous space.  The evader follows one of several
fixed strategies, configurable below.

Based on the experimental setup from:
  "Adaptive Adversarial Agents via NEAT: Evolving Pursuit Behaviour
   Against Non-Stationary Player Strategies" (Aryan Jha, 2026)
"""

import multiprocessing
import os
import pickle

import neat
import pursuit_evasion
import visualize

runs_per_genome = 5
evader_strategy = 'random'


def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = pursuit_evasion.PursuitEvasion(evader_strategy=evader_strategy)

    total_reward = 0.0
    for _ in range(runs_per_genome):
        state = env.step([0.0, 0.0])[0]
        ep_reward = 0.0
        while True:
            action = net.activate(state)
            state, reward, done = env.step(action)
            ep_reward += reward
            if done:
                break
        total_reward += ep_reward

    return total_reward / (runs_per_genome * env.max_steps)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(50))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate, 100)

    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(f'\nBest genome:\n{winner!s}')

    visualize.plot_stats(stats, ylog=False, view=True, filename="pursuit-fitness.svg")
    visualize.plot_species(stats, view=True, filename="pursuit-speciation.svg")

    node_names = {
        -1: 'rel_x', -2: 'rel_y',
        -3: 'evader_vx', -4: 'evader_vy',
        -5: 'evader_ax', -6: 'evader_ay',
        -7: 'pursuer_vx', -8: 'pursuer_vy',
        0: 'Δvx', 1: 'Δvy',
    }
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-pruned.gv", prune_unused=True)


if __name__ == '__main__':
    run()
