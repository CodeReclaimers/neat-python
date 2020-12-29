"""

    [Hoverboard]
    A 2D game consisting of a hoverboard with two thrusters. You can control their thrust individually.
    The challenge is to keep the platform floating in the middle of the screen.
    If it touches any of the walls, it's destroyed.

    <How to Run>
    > pip install pygame
    > python hoverboard.py
    > Have a little fun with it
    > Get frustrated
    > Go check the NEAT-NSGA2 controller

    <How to Play>
    Q : ++ left thruster
    A : -- left thruster
    P : ++ right thruster
    L : -- right thruster

    @author: Hugo Aboud (@hugoaboud)

"""

import math
import sys
import pygame
from pygame.locals import QUIT

"""
    Settings
"""

# Dimensions
DISPLAY = (512,512) # w, h
THRUSTER = (20,300,10) # w, h, padding

# Colors
COLOR_BACKGROUND = (30,30,30)
COLOR_PLAYER = (180,200,255)
COLOR_THRUSTER_OFF = (50,50,50)
COLOR_THRUSTER_LEFT = (200,0,127)
COLOR_THRUSTER_RIGHT = (0,127,200)

# Physics
MASS = 2
GRAVITY = 1.5
FORCE = 3
INTERTIA_MOMENTUM = 0.05

# Input
INPUT_STEP = 0.1

##
#   Hoverboard
#   Most of the Physics and Rendering is happening here
##

class Hoverboard:
    def __init__(self, x = 0.5, y = 0.5, w = 100, h = 10, angle = 0, velocity = None, ang_velocity = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle
        self.velocity = velocity if (velocity != None) else [0,0]
        self.ang_velocity = ang_velocity

        # calculate normal
        rad = math.pi*self.angle/180
        self.normal = (-math.sin(rad), -math.cos(rad))

        # default thrust (last is used for smoother display)
        self.last_thrust = [0.5,0.5]
        self.thrust = [0.5,0.5]

        # create surface to draw the hoverboard into
        self.surface = pygame.Surface((w,h*5), pygame.SRCALPHA)

    def render(self, screen):
        # clear surface
        self.surface.fill((0,0,0,0))
        # draw platform (rect)
        pygame.draw.rect(self.surface, COLOR_PLAYER, (0, self.h*2, self.w, self.h))
        # draw thruster left (pixel)
        l = (self.thrust[0]+self.last_thrust[0])/2.0
        th = 2*self.h*l
        for y in range(int(th)):
            for x in range(self.h):
                self.surface.set_at((x,self.h*3+y),
                                (COLOR_THRUSTER_LEFT[0],
                                 COLOR_THRUSTER_LEFT[1],
                                 COLOR_THRUSTER_LEFT[2],
                                 int(255*(1-y/th))))
        # draw thruster right (pixel)
        r = (self.thrust[1]+self.last_thrust[1])/2.0
        th = 2*self.h*r
        for y in range(int(th)):
            for x in range(self.w-self.h,self.w):
                self.surface.set_at((x,self.h*3+y),
                                (COLOR_THRUSTER_RIGHT[0],
                                 COLOR_THRUSTER_RIGHT[1],
                                 COLOR_THRUSTER_RIGHT[2],
                                 int(255*(1-y/th))))
        # rotate surface around center
        rotated = pygame.transform.rotate(self.surface, self.angle)
        rect = rotated.get_rect()
        rect[0] -= rect[2]/2
        rect[1] -= rect[3]/2
        # position and blit
        rect[0] += self.x*DISPLAY[0]
        rect[1] += self.y*DISPLAY[1]
        screen.blit(rotated, rect)

    # update physics
    def update(self, delta_t):
        # gravity
        # > increases Y velocity
        self.velocity[1] += GRAVITY*delta_t

        # thrust torque
        # > torque to angular acceleration (no radius used, ajusted by inerta momentum)
        # > angular acceleration +> angular velocity
        ang_accel_l = (FORCE/INTERTIA_MOMENTUM)*self.thrust[0]
        ang_accel_r = (FORCE/INTERTIA_MOMENTUM)*self.thrust[1]
        self.ang_velocity += (ang_accel_r-ang_accel_l)*delta_t

        # (TODO: drag / air resistance )

        # thrust force
        # > force to linear acceleration
        # > linear acceleration +> linear velocity
        accel = (FORCE/MASS)*(self.thrust[0]+self.thrust[1])
        self.velocity[0] += self.normal[0]*accel*delta_t
        self.velocity[1] += self.normal[1]*accel*delta_t

        # update position and angle
        self.x += self.velocity[0]*delta_t
        self.y += self.velocity[1]*delta_t
        self.angle += self.ang_velocity*delta_t
        # update normal
        rad = math.pi*self.angle/180
        self.normal = (-math.sin(rad), -math.cos(rad))

    # update thrust values within [0,1] bounds
    # also keep last value for smoother display
    def set_thrust(self, left, right):
        self.last_thrust = list(self.thrust)
        self.thrust[0] = left
        if (self.thrust[0] < 0): self.thrust[0] = 0
        elif (self.thrust[0] > 1): self.thrust[0] = 1
        self.thrust[1] = right
        if (self.thrust[1] < 0): self.thrust[1] = 0
        elif (self.thrust[1] > 1): self.thrust[1] = 1

##
#   Game
#   Here's the game loop, main rendering method, input handling and
#   construction/destruction methods.
##

class Game:
    def __init__(self, start_angle = 0, frontend = True, network_gui = None):
        self.start_angle = start_angle
        self.frontend = frontend

        # (optional)
        # use a NeuralNetworkGUI to display real time genome topology and info
        self.network_gui = network_gui

        # reset flag is set when the hoverboard collides with the borders
        # it is true after the update when the collision happens, and before the next update
        self.reset_flag = False

        # create hoverboard
        self.hoverboard = Hoverboard(angle = start_angle)

        # initialize pygame modules
        pygame.init()

        # if Front-End enabled, open pygame window
        if (self.frontend):
            self.screen = pygame.display.set_mode(DISPLAY)

    # main render
    def render(self):
        # clear screen
        self.screen.fill(COLOR_BACKGROUND)

        # render hoverboard
        self.hoverboard.render(self.screen)

        # render thrusters UI (left and right bars)
        l = (self.hoverboard.thrust[0]+self.hoverboard.last_thrust[0])/2.0 # smooth display
        r = (self.hoverboard.thrust[1]+self.hoverboard.last_thrust[1])/2.0 # smooth display
        pygame.draw.rect(self.screen, COLOR_THRUSTER_OFF, (THRUSTER[2], (DISPLAY[1]-THRUSTER[1])/2,THRUSTER[0],THRUSTER[1]*(1-l)))
        pygame.draw.rect(self.screen, COLOR_THRUSTER_LEFT, (THRUSTER[2], (DISPLAY[1]-THRUSTER[1])/2+THRUSTER[1]*(1-l),THRUSTER[0],THRUSTER[1]*l))
        pygame.draw.rect(self.screen, COLOR_THRUSTER_OFF, (DISPLAY[0]-THRUSTER[0]-THRUSTER[2], (DISPLAY[1]-THRUSTER[1])/2,THRUSTER[0],THRUSTER[1]*(1-r)))
        pygame.draw.rect(self.screen, COLOR_THRUSTER_RIGHT, (DISPLAY[0]-THRUSTER[0]-THRUSTER[2], (DISPLAY[1]-THRUSTER[1])/2+THRUSTER[1]*(1-r),THRUSTER[0],THRUSTER[1]*r))

        # (optional) render NeuralNetworkGUI
        if (self.network_gui != None):
            self.network_gui.render(self.screen)

    # game loop
    def loop(self):
        last_t = pygame.time.get_ticks()
        while True:
            t = pygame.time.get_ticks()
            # Events
            for event in pygame.event.get():
                # Input Handling
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.hoverboard.thrust[0] += INPUT_STEP
                        if (self.hoverboard.thrust[0] > 1): self.hoverboard.thrust[0] = 1
                    if event.key == pygame.K_a:
                        self.hoverboard.thrust[0] -= INPUT_STEP
                        if (self.hoverboard.thrust[0] < 0): self.hoverboard.thrust[0] = 0
                    if event.key == pygame.K_p:
                        self.hoverboard.thrust[1] += INPUT_STEP
                        if (self.hoverboard.thrust[1] > 1): self.hoverboard.thrust[1] = 1
                    if event.key == pygame.K_l:
                        self.hoverboard.thrust[1] -= INPUT_STEP
                        if (self.hoverboard.thrust[1] < 0): self.hoverboard.thrust[1] = 0
                # Quit
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            # Update
            self.update((t-last_t)/1000)
            last_t = t

    # update physics, collisions and game state
    # this is separated from the game loop so it can be called from an outside loop
    # this way, it's possible to control the game loop externally to evaluate the genomes
    def update(self, delta_t):
        # Clear reset flag
        if (self.reset_flag): self.reset_flag = False
        # Update hoverboard physics
        self.hoverboard.update(delta_t)
        # Collision (end game)
        if (self.hoverboard.x <= 0 or self.hoverboard.x >= 1 or
            self.hoverboard.y <= 0 or self.hoverboard.y >= 1):
            self.reset()
        # Front-End (Render)
        if (self.frontend):
            self.render()
            pygame.display.update()

    # reset the hoverboard to the initial state
    def reset(self):
        self.hoverboard = Hoverboard(angle = self.start_angle)
        # Set reset flag
        self.reset_flag = True

##
#   If you run this file with `python hoverboard.py`, it creates a Game instance
#   and runs the `loop` method. This way, the game can also be played without NEAT.
##
if __name__ == "__main__":
   Game().loop()
