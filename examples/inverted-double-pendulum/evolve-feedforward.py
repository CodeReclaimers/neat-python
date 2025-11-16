"""
Evolve a control network for the Gymnasium InvertedDoublePendulum-v5 environment.
The inverted double pendulum has two poles connected serially and mounted on a cart.
The goal is to balance both poles by applying forces to the cart.
"""

import multiprocessing
import os
import pickle

import gymnasium as gym
import neat
import visualize

# Environment parameters
runs_per_net = 3
max_steps = 1000


def eval_genome(genome, config):
    """
    Evaluates a genome by testing it on the inverted double pendulum environment.
    
    Returns the average fitness across multiple runs.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    fitnesses = []
    
    for _ in range(runs_per_net):
        env = gym.make('InvertedDoublePendulum-v5')
        observation, info = env.reset()
        
        fitness = 0.0
        for step in range(max_steps):
            # Get action from neural network
            action = net.activate(observation)
            
            # Step the environment
            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward
            
            if terminated or truncated:
                break
        
        env.close()
        fitnesses.append(fitness)
    
    # Return average fitness across all runs
    return sum(fitnesses) / len(fitnesses)


def eval_genomes(genomes, config):
    """
    Evaluates all genomes in the population.
    """
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(config_file):
    """
    Runs the NEAT algorithm to evolve a controller for the inverted double pendulum.
    """
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)

    # Create the population
    pop = neat.Population(config)

    # Add reporters to track progress
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(10))

    # Run evolution with parallel evaluation
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate, 1000)

    # Save the winner
    with open('winner-feedforward.pickle', 'wb') as f:
        pickle.dump(winner, f)

    print('\n\nBest genome:\n{!s}'.format(winner))

    # Visualize the results
    visualize.plot_stats(stats, ylog=False, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    # Create node names for visualization
    node_names = {
        -1: 'x', -2: 'y', -3: 'z',
        -4: 'θ1', -5: 'θ2', -6: 'ẋ',
        -7: 'ẏ', -8: 'ż', -9: 'v_tip',
        0: 'force'
    }
    
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                      filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                      filename="winner-feedforward-pruned.gv", prune_unused=True)

    return winner, stats


if __name__ == '__main__':
    # Determine path to configuration file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    
    winner, stats = run(config_path)
