'''
General settings and implementation of the single-pole cart system dynamics.
'''

from math import cos, pi, sin
import random

position_limit = 2.4
angle_limit_radians = 45 * pi / 180
num_steps = 10 ** 5


class CartPole(object):
    gravity = 9.8  # acceleration due to gravity, positive is downward, m/sec^2
    mcart = 1.0  # cart mass in kg
    mpole = 0.1  # pole mass in kg
    lpole = 0.5  # half the pole length in meters
    time_step = 0.02  # time step in seconds

    def __init__(self, x=None, theta=None, dx=None, dtheta=None):
        if x is None:
            x = random.uniform(-0.8 * position_limit, 0.8 * position_limit)

        if theta is None:
            theta = random.uniform(-0.8 * angle_limit_radians, 0.8 * angle_limit_radians)

        if dx is None:
            dx = random.uniform(-1.0, 1.0)

        if dtheta is None:
            dtheta = random.uniform(-1.0, 1.0)

        self.t = 0.0
        self.x = x
        self.theta = theta

        self.dx = dx
        self.dtheta = dtheta

        self.xacc = 0.0
        self.tacc = 0.0

    def step(self, force):
        '''
        Update the system state using leapfrog integration.
            x_{i+1} = x_i + v_i * dt + 0.5 * a_i * dt^2
            v_{i+1} = v_i + 0.5 * (a_i + a_{i+1}) * dt
        '''
        # Locals for readability.
        g = self.gravity
        mp = self.mpole
        mc = self.mcart
        mt = mp + mc
        L = self.lpole
        dt = self.time_step

        # Remember acceleration from previous step.
        tacc0 = self.tacc
        xacc0 = self.xacc

        # Update position/angle.
        self.x += dt * self.dx + 0.5 * xacc0 * dt ** 2
        self.theta += dt * self.dtheta + 0.5 * tacc0 * dt ** 2

        # Compute new accelerations as given in "Correct equations for the dynamics of the cart-pole system"
        # by Razvan V. Florian.
        st = sin(self.theta)
        ct = cos(self.theta)
        tacc1 = (g * st + ct * (-force - mp * L * self.dtheta ** 2 * st) / mt) / (L * (4.0 / 3 - mp * ct ** 2 / mt))
        xacc1 = (force + mp * L * (self.dtheta ** 2 * st - tacc1 * ct)) / mt

        # Update velocities.
        self.dx += 0.5 * (xacc0 + xacc1) * dt
        self.dtheta += 0.5 * (tacc0 + tacc1) * dt

        # Remember current acceleration for next step.
        self.tacc = tacc1
        self.xacc = xacc1
        self.t += dt

    def get_scaled_state(self):
        '''Get full state, scaled into (approximately) [0, 1].'''
        return [0.5 * (self.x + position_limit) / position_limit,
                (self.dx + 0.75) / 1.5,
                0.5 * (self.theta + angle_limit_radians) / angle_limit_radians,
                (self.dtheta + 1.0) / 2.0]


def discrete_actuator_force(action):
    return 10.0 if action[0] > 0.5 else -10.0


def noisy_actuator_force(action):
    a = action[0] + random.uniform(-0.2, 0.2)
    return 10.0 if a > 0.5 else -10.0


def run_simulation(sim, net, force_func):
    '''
    Run the given simulation for up to num_steps time steps.
    Returns the number of time steps during which the position and angle were within limits.
    '''
    for trials in xrange(num_steps):
        inputs = sim.get_scaled_state()
        action = net.parallel_activate(inputs)

        # Apply action to the simulated cart-pole
        force = force_func(action)
        sim.step(force)

        # Stop if the network fails to keep the cart within the position or angle limits.
        # The per-run fitness is the number of time steps the network can balance the pole
        # without exceeding these limits.
        if abs(sim.x) >= position_limit or abs(sim.theta) >= angle_limit_radians:
            return trials

    return num_steps
