## NSGA-II examples ##

The scripts in this directory show examples of using NEAT to control a hoverboard on a game.
It uses CTRNNs to control the intensity of both left and right thrusters of the hoverboard, based on it's velocity, angular velocity and normal vector. All those informations could be retreived from real world sensors.

![hoverboard-reference](https://i.imgur.com/SfPblbG.gif)

#### Play the Game

```
pip install pygame
python hoverboard.py
```

- Q/A : +/- left thruster
- P/L : +/- right thruster

#### Reference (without NSGA-II)

A reference example uses the Default Reproduction method, with a single value of fitness: runtime.

The evolution converges fast on simple behaviours such as overcoming gravity by boosting both thrusters simultaneously, however a more refined fitness method should include the total variation of velocities and normal vector to help it converge faster to a stable controller.

```
pip install pygame
python evolve-reference.py
```

#### NSGA-II

TODO: The NSGA-II method uses multiple fitness values to pick the best performing genomes of each species.

#### Visualize

TODO: plot checkpoints report with pyplot and watch the simulation.
