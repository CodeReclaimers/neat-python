## Pursuit-Evasion Example ##

This example evolves a feed-forward neural network controller for a pursuer agent in a 2D continuous pursuit-evasion game.  The evader follows one of several fixed strategies, and the pursuer must learn to capture it through vector-based control.

The environment and experimental setup are based on the paper:

> **Adaptive Adversarial Agents via NEAT: Evolving Pursuit Behaviour Against Non-Stationary Player Strategies** (Aryan Jha, 2026)

### Environment ###

- Continuous 2D space of 800 × 800 units.
- **Observation (8-dim):** relative position (Δx, Δy), evader velocity (vx, vy), evader acceleration (ax, ay), pursuer velocity (vx, vy).
- **Action (2-dim):** velocity change [Δvx, Δvy] clipped to [-1, 1], scaled internally by 2.0.
- **Reward:** `+1` on capture, `-distance/800` per step otherwise.
- **Episode:** 500 steps max.
- **Capture condition:** pursuer within 20 units of evader.

### Evader Strategies ###

The evader can follow one of four strategies (set via `evader_strategy` in `evolve-feedforward.py`):

| Strategy | Behaviour |
|---|---|
| `random` | Random bounded acceleration (default). |
| `flee` | Moves directly away from pursuer at max speed. |
| `dodge` | Predicts pursuer trajectory via velocity extrapolation and moves perpendicular. |
| `oscillate` | Follows sinusoidal motion patterns. |

### Running ###

```bash
# Train a pursuer against the random evader (default)
python evolve-feedforward.py

# To change the evader strategy, edit the `evader_strategy` variable in the script.
```

### Files ###

* `evolve-feedforward.py` — Main evolution script.
* `pursuit_evasion.py` — The pursuit-evasion environment.
* `config-feedforward` — NEAT configuration (based on the paper's setup: pop 80, 8 inputs, 2 outputs).
* `visualize.py` — Plotting and network visualization utilities.
