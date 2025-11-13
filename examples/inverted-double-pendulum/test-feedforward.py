"""
Test and visualize a trained controller for the inverted double pendulum.
"""

import os
import pickle

import gymnasium as gym
import neat


def test_network(net, episodes=10, render=True, camera_distance=4.0):
    """
    Tests a neural network controller on the inverted double pendulum.
    
    Args:
        net: The neural network to test
        episodes: Number of episodes to run
        render: Whether to render the environment
        camera_distance: Distance of camera from the pendulum (higher = more zoomed out)
    """
    fitnesses = []
    
    # Create environment once and reuse it for all episodes
    if render:
        env = gym.make('InvertedDoublePendulum-v5', render_mode='human')
    else:
        env = gym.make('InvertedDoublePendulum-v5')
    
    try:
        for episode in range(episodes):
            observation, info = env.reset()
            
            # Adjust camera distance for better view (only needs to be set once)
            if episode == 0 and render and hasattr(env.unwrapped, 'mujoco_renderer'):
                renderer = env.unwrapped.mujoco_renderer
                if renderer.viewer is not None:
                    renderer.viewer.cam.distance = camera_distance
            
            fitness = 0.0
            step = 0
            
            while True:
                step += 1
                # Get action from network
                action = net.activate(observation)
                
                # Step environment
                observation, reward, terminated, truncated, info = env.step(action)
                fitness += reward
                
                if terminated or truncated:
                    break
            
            fitnesses.append(fitness)
            print(f"Episode {episode + 1}: steps={step}, fitness={fitness:.2f}")
    
    finally:
        env.close()
    
    avg_fitness = sum(fitnesses) / len(fitnesses)
    max_fitness = max(fitnesses)
    min_fitness = min(fitnesses)
    
    print(f"\nResults over {episodes} episodes:")
    print(f"  Average fitness: {avg_fitness:.2f}")
    print(f"  Max fitness: {max_fitness:.2f}")
    print(f"  Min fitness: {min_fitness:.2f}")
    
    return fitnesses


def load_and_test(genome_path, config_path, episodes=10, render=True, camera_distance=4.0):
    """
    Loads a saved genome and tests it.
    
    Args:
        genome_path: Path to the pickled genome file
        config_path: Path to the NEAT config file
        episodes: Number of test episodes
        render: Whether to render the environment
        camera_distance: Distance of camera from the pendulum (higher = more zoomed out)
    """
    # Load the config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # Load the genome
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    print('Loaded genome:')
    print(genome)
    
    # Create the network
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Test the network
    return test_network(net, episodes=episodes, render=render, camera_distance=camera_distance)


if __name__ == '__main__':
    import sys
    
    # Determine paths
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    
    # Check if a genome file was specified
    if len(sys.argv) > 1:
        genome_path = sys.argv[1]
    else:
        genome_path = os.path.join(local_dir, 'winner-feedforward.pickle')
    
    # Check if genome file exists
    if not os.path.exists(genome_path):
        print(f"Error: Genome file not found at {genome_path}")
        print("Please train a network first by running evolve-feedforward.py")
        sys.exit(1)
    
    # Test the network
    print(f"Testing genome from: {genome_path}\n")
    load_and_test(genome_path, config_path, episodes=5, render=True, camera_distance=4.0)
