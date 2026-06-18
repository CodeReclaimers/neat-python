"""
2D pursuit-evasion environment for NEAT experiments.

A pursuer (controlled by a NEAT-evolved network) chases an evader
in a bounded continuous space.  The evader follows one of several
fixed strategies, and the pursuer must learn to capture it.

This environment is based on the one described in:
  "Adaptive Adversarial Agents via NEAT: Evolving Pursuit Behaviour
   Against Non-Stationary Player Strategies" (Aryan Jha, 2026)
"""

import math
import random

ARENA_SIZE = 400.0
CAPTURE_RADIUS = 20.0
MAX_STEPS = 500
MAX_SPEED = 5.0
ACTION_SCALE = 2.0


class PursuitEvasion:
    def __init__(self, evader_strategy='random'):
        self.arena_size = ARENA_SIZE
        self.capture_radius = CAPTURE_RADIUS
        self.max_steps = MAX_STEPS
        self.evader_strategy = evader_strategy

        self.step_count = 0
        self.captured = False

        margin = 50.0
        self.pursuer_pos = [random.uniform(margin, 2 * ARENA_SIZE - margin),
                            random.uniform(margin, 2 * ARENA_SIZE - margin)]
        self.pursuer_vel = [0.0, 0.0]

        while True:
            ex = random.uniform(margin, 2 * ARENA_SIZE - margin)
            ey = random.uniform(margin, 2 * ARENA_SIZE - margin)
            dx = ex - self.pursuer_pos[0]
            dy = ey - self.pursuer_pos[1]
            if math.hypot(dx, dy) > self.capture_radius * 3:
                break
        self.evader_pos = [ex, ey]
        self.evader_vel = [0.0, 0.0]
        self.evader_prev_vel = [0.0, 0.0]
        self.evader_phase = random.uniform(0, 2 * math.pi)

    def _clamp(self, pos):
        pos[0] = max(0.0, min(2 * self.arena_size, pos[0]))
        pos[1] = max(0.0, min(2 * self.arena_size, pos[1]))

    def _evader_action(self):
        if self.evader_strategy == 'random':
            return [random.uniform(-1, 1) * 3.0,
                    random.uniform(-1, 1) * 3.0]
        elif self.evader_strategy == 'flee':
            dx = self.evader_pos[0] - self.pursuer_pos[0]
            dy = self.evader_pos[1] - self.pursuer_pos[1]
            dist = math.hypot(dx, dy)
            if dist > 0:
                return [dx / dist * 5.0, dy / dist * 5.0]
        elif self.evader_strategy == 'dodge':
            pred_px = self.pursuer_pos[0] + self.pursuer_vel[0] * 10
            pred_py = self.pursuer_pos[1] + self.pursuer_vel[1] * 10
            dx = self.evader_pos[0] - pred_px
            dy = self.evader_pos[1] - pred_py
            perp = [-dy, dx]
            dist = math.hypot(*perp)
            if dist > 0:
                return [perp[0] / dist * 5.0, perp[1] / dist * 5.0]
        elif self.evader_strategy == 'oscillate':
            t = self.step_count / 30.0
            return [math.cos(t) * 4.0, math.sin(t) * 4.0]
        return [0.0, 0.0]

    def step(self, action):
        if self.captured:
            return self._get_state(), 0.0, True

        ax = max(-1.0, min(1.0, action[0]))
        ay = max(-1.0, min(1.0, action[1]))

        self.pursuer_vel[0] += ax * ACTION_SCALE
        self.pursuer_vel[1] += ay * ACTION_SCALE
        self.pursuer_vel[0] = max(-MAX_SPEED, min(MAX_SPEED, self.pursuer_vel[0]))
        self.pursuer_vel[1] = max(-MAX_SPEED, min(MAX_SPEED, self.pursuer_vel[1]))

        self.evader_prev_vel = list(self.evader_vel)
        evader_accel = self._evader_action()
        self.evader_vel[0] += evader_accel[0]
        self.evader_vel[1] += evader_accel[1]
        self.evader_vel[0] = max(-MAX_SPEED, min(MAX_SPEED, self.evader_vel[0]))
        self.evader_vel[1] = max(-MAX_SPEED, min(MAX_SPEED, self.evader_vel[1]))

        for i in range(2):
            self.pursuer_pos[i] += self.pursuer_vel[i]
            self.evader_pos[i] += self.evader_vel[i]
        self._clamp(self.pursuer_pos)
        self._clamp(self.evader_pos)

        self.step_count += 1

        dx = self.pursuer_pos[0] - self.evader_pos[0]
        dy = self.pursuer_pos[1] - self.evader_pos[1]
        dist = math.hypot(dx, dy)

        self.captured = dist < self.capture_radius
        done = self.captured or (self.step_count >= self.max_steps)
        reward = 1.0 if self.captured else -dist / (2 * self.arena_size)

        return self._get_state(), reward, done

    def _get_state(self):
        rel_x = (self.evader_pos[0] - self.pursuer_pos[0]) / (2 * self.arena_size)
        rel_y = (self.evader_pos[1] - self.pursuer_pos[1]) / (2 * self.arena_size)
        evader_accel_x = self.evader_vel[0] - self.evader_prev_vel[0]
        evader_accel_y = self.evader_vel[1] - self.evader_prev_vel[1]
        return [
            rel_x,
            rel_y,
            self.evader_vel[0] / MAX_SPEED,
            self.evader_vel[1] / MAX_SPEED,
            evader_accel_x / (ACTION_SCALE * 2),
            evader_accel_y / (ACTION_SCALE * 2),
            self.pursuer_vel[0] / MAX_SPEED,
            self.pursuer_vel[1] / MAX_SPEED,
        ]

    def get_pursuer_pos(self):
        return tuple(self.pursuer_pos)

    def get_evader_pos(self):
        return tuple(self.evader_pos)

    def get_capture_time(self):
        return self.step_count if self.captured else None
