#!/usr/bin/env python3
"""
Check if all required dependencies are installed for the Hopper-v5 example.
"""

import sys


def check_dependencies():
    """Check if all required packages are available."""
    missing = []

    # Check core dependencies
    try:
        import neat  # noqa: F401
        print("\u2713 neat-python is installed")
    except ImportError:
        print("\u2717 neat-python is NOT installed")
        missing.append("neat-python")

    try:
        import gymnasium  # noqa: F401
        print("\u2713 gymnasium is installed")
    except ImportError:
        print("\u2717 gymnasium is NOT installed")
        missing.append("gymnasium")

    # Check optional dependencies
    try:
        import pygame  # noqa: F401
        print("\u2713 pygame is installed (for rendering)")
    except ImportError:
        print("\u26a0 pygame is NOT installed (optional, needed for visualization)")

    try:
        import graphviz  # noqa: F401
        print("\u2713 graphviz is installed (for network visualization)")
    except ImportError:
        print("\u26a0 graphviz is NOT installed (optional, needed for network diagrams)")

    try:
        import matplotlib  # noqa: F401
        print("\u2713 matplotlib is installed (for plotting)")
    except ImportError:
        print("\u26a0 matplotlib is NOT installed (optional, needed for fitness plots)")

    # Check if Hopper-v5 environment is available
    if not missing or "gymnasium" not in missing:
        try:
            import gymnasium as gym

            env = gym.make("Hopper-v5")
            obs_space = env.observation_space
            action_space = env.action_space
            env.close()
            print("\u2713 Hopper-v5 environment is available")
            print(f"  - Observation space: {obs_space}")
            print(f"  - Action space: {action_space}")
        except Exception as e:  # pragma: no cover - environment-specific
            print(f"\u2717 Hopper-v5 environment error: {e}")
            missing.append("mujoco (Hopper requires MuJoCo)")

    print("\n" + "=" * 60)
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
