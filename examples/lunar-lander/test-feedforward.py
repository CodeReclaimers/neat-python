"""\
Test and visualize the performance of the best genome produced by
examples/lunar-lander/evolve-feedforward.py on the LunarLander-v3 environment.
"""

import os
import pickle
import sys

import gymnasium as gym
import neat


def run_episodes(net, episodes=3, render=True):
    """Run a few episodes using the provided network and optionally render."""
    if render:
        env = gym.make("LunarLander-v3", render_mode="human")
    else:
        env = gym.make("LunarLander-v3")

    try:
        rewards = []
        for episode in range(episodes):
            observation, info = env.reset()
            total_reward = 0.0
            step = 0

            while True:
                step += 1
                action_values = net.activate(observation)
                action = max(range(len(action_values)), key=lambda i: action_values[i])

                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            print(
                f"Episode {episode + 1}: steps={step}, total_reward={total_reward:.2f}",
            )
    finally:
        env.close()

    if rewards:
        avg = sum(rewards) / len(rewards)
        print(f"\nAverage reward over {len(rewards)} episodes: {avg:.2f}")


def load_and_test(genome_path, config_path, episodes=3, render=True):
    """Load a saved genome and test it on LunarLander-v3."""
    # Load the config.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Load the genome.
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    print("Loaded genome:")
    print(genome)

    # Create the network and run episodes.
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    run_episodes(net, episodes=episodes, render=render)


if __name__ == "__main__":
    # Determine local paths.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward")

    # Optional argument: custom path to winner genome.
    if len(sys.argv) > 1:
        genome_path = sys.argv[1]
    else:
        genome_path = os.path.join(local_dir, "winner-feedforward.pickle")

    if not os.path.exists(genome_path):
        print(f"Error: Genome file not found at {genome_path}")
        print("Please train a network first by running evolve-feedforward.py")
        sys.exit(1)

    print(f"Testing genome from: {genome_path}\n")
    load_and_test(genome_path, config_path, episodes=3, render=True)
