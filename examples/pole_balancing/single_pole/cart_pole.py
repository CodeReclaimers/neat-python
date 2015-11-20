from math import cos, pi, sin
from neat.math_util import randrange

position_limit = 2.4
angle_limit_radians = 45 * pi / 180


class CartPole(object):
    gravity = 9.8  # acceleration due to gravity, positive is downward, m/sec^2
    mcart = 1.0  # cart mass in kg
    mpole = 0.1  # pole mass in kg
    lpole = 0.5  # half the pole length in meters
    time_step = 0.02  # time step in seconds

    def __init__(self, x=None, theta=None, dx=None, dtheta=None):
        if x is None:
            x = randrange(-position_limit, position_limit)

        if theta is None:
            theta = randrange(-0.2, 0.2)

        if dx is None:
            dx = randrange(-1.0, 1.0)

        if dtheta is None:
            dtheta = randrange(-1.5, 1.5)

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


