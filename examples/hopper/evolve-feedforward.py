"""
Evolve a control network for the Gymnasium Hopper-v5 environment.

The Hopper is a two-dimensional one-legged robot. The goal is to hop
forward as far as possible without falling.
"""

import multiprocessing
import os
import pickle

import gymnasium as gym
import neat
import visualize

# Evaluation parameters
runs_per_net = 3
max_steps = 1000


def eval_genome(genome, config):
    """Evaluate a single genome on the Hopper-v5 environment.

    Returns the average fitness over multiple runs.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []

    for _ in range(runs_per_net):
        # Create a fresh environment for each run (no rendering during training).
        env = gym.make("Hopper-v5")
        observation, info = env.reset()

        total_reward = 0.0
        for _ in range(max_steps):
            # Network outputs three continuous action values in [-1, 1].
            action = net.activate(observation)

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        env.close()
        fitnesses.append(total_reward)

    # Use the average reward over runs as the fitness.
    return sum(fitnesses) / len(fitnesses)


def eval_genomes(genomes, config):
    """Evaluate all genomes in the population."""
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(config_file):
    """Run NEAT to evolve a controller for Hopper-v5."""
    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # Periodic checkpoints, similar to other examples.
    p.add_reporter(neat.Checkpointer(10))

    # Use parallel evaluation across available CPU cores.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    # Run until solution or fitness threshold is reached (see config).
    winner = p.run(pe.evaluate, 300)

    # Display the winning genome.
    print(f"\nBest genome:\n{winner!s}")

    # Save the winner for later reuse in test-feedforward.py.
    with open("winner-feedforward.pickle", "wb") as f:
        pickle.dump(winner, f)

    # Fitness & species plots (analogous to other examples).
    visualize.plot_stats(
        stats,
        ylog=False,
        view=True,
        filename="feedforward-fitness.svg",
    )
    visualize.plot_species(
        stats,
        view=True,
        filename="feedforward-speciation.svg",
    )

    # Node labels for easier interpretation of the evolved controller.
    # Hopper-v5 observations form an 11D vector including torso height,
    # torso angle, joint angles, and their velocities. We give a few
    # illustrative labels here for readability.
    node_names = {
        -1: "z",
        -2: "theta",
        -3: "thigh_angle",
        -4: "leg_angle",
        -5: "foot_angle",
        -6: "z_vel",
        -7: "theta_vel",
        -8: "thigh_vel",
        -9: "leg_vel",
        -10: "foot_vel",
        -11: "x_vel",
        0: "thigh_torque",
        1: "leg_torque",
        2: "foot_torque",
    }

    # Full and pruned network diagrams for the winning genome.
    visualize.draw_net(
        config,
        winner,
        view=True,
        node_names=node_names,
        filename="winner-feedforward.gv",
    )
    visualize.draw_net(
        config,
        winner,
        view=True,
        node_names=node_names,
        filename="winner-feedforward-pruned.gv",
        prune_unused=True,
    )

    return winner, stats


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward")
    run(config_path)
