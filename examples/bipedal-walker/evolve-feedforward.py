"""\
Feed-forward BipedalWalker-v3 control example using Gymnasium.

This example is structured similarly to examples/lunar-lander/evolve-feedforward.py and
produces the same kinds of visual artifacts:

* Fitness curve over generations
* Species size stack plot
* Network diagrams (full and pruned) of the winning genome
"""

import multiprocessing
import os
import pickle

import gymnasium as gym
import neat
import visualize

# Evaluation parameters.
runs_per_net = 1
max_steps = 2000


def eval_genome(genome, config):
    """Evaluate a single genome on the BipedalWalker-v3 environment."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []

    for _ in range(runs_per_net):
        # Create a fresh environment for each run (no rendering during training).
        env = gym.make("BipedalWalker-v3")
        observation, info = env.reset()

        total_reward = 0.0
        for _ in range(max_steps):
            # Network outputs four continuous action values in [-1, 1].
            # With tanh activations (see config), the raw outputs are already
            # in a good range for the BipedalWalker action space.
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
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(config_file):
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

    # Visualization artifacts analogous to examples/xor/evolve-feedforward.py.
    # Fitness & species plots.
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
    # BipedalWalker-v3 observations are a 24-dimensional vector that includes
    # hull angle/velocity, joint angles/velocities, leg contact, and LIDAR
    # measurements. For brevity, we group them into coarse labels here.
    node_names = {
        # Example grouping of observation components (indices are illustrative):
        -1: "hull_angle",
        -2: "hull_ang_vel",
        -3: "vel_x",
        -4: "vel_y",
        -5: "hip_1",
        -6: "knee_1",
        -7: "hip_2",
        -8: "knee_2",
        # Remaining inputs (-9 .. -24) are left unnamed for clarity.
        0: "motor_hip_1",
        1: "motor_knee_1",
        2: "motor_hip_2",
        3: "motor_knee_2",
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
