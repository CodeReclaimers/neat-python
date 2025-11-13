#!/usr/bin/env python3
"""
Check if all required dependencies are installed for the inverted double pendulum example.
"""

import sys

def check_dependencies():
    """Check if all required packages are available."""
    missing = []
    
    # Check core dependencies
    try:
        import neat
        print("✓ neat-python is installed")
    except ImportError:
        print("✗ neat-python is NOT installed")
        missing.append("neat-python")
    
    try:
        import gymnasium
        print("✓ gymnasium is installed")
    except ImportError:
        print("✗ gymnasium is NOT installed")
        missing.append("gymnasium")
    
    # Check optional dependencies
    try:
        import pygame
        print("✓ pygame is installed (for rendering)")
    except ImportError:
        print("⚠ pygame is NOT installed (optional, needed for visualization)")
    
    try:
        import graphviz
        print("✓ graphviz is installed (for network visualization)")
    except ImportError:
        print("⚠ graphviz is NOT installed (optional, needed for network diagrams)")
    
    try:
        import matplotlib
        print("✓ matplotlib is installed (for plotting)")
    except ImportError:
        print("⚠ matplotlib is NOT installed (optional, needed for fitness plots)")
    
    # Check if InvertedDoublePendulum environment is available
    if not missing or "gymnasium" not in missing:
        try:
            import gymnasium as gym
            env = gym.make('InvertedDoublePendulum-v5')
            obs_space = env.observation_space
            action_space = env.action_space
            env.close()
            print(f"✓ InvertedDoublePendulum-v5 environment is available")
            print(f"  - Observation space: {obs_space}")
            print(f"  - Action space: {action_space}")
        except Exception as e:
            print(f"✗ InvertedDoublePendulum-v5 environment error: {e}")
            missing.append("mujoco (InvertedDoublePendulum requires MuJoCo)")
    
    print("\n" + "="*60)
    if missing:
        print("MISSING REQUIRED DEPENDENCIES:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    else:
        print("All required dependencies are installed!")
        print("You can start training with: python evolve-feedforward.py")
        return True

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)
