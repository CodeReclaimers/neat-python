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

    author: @hugoaboud

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
INTERTIA_MOMENTUM = 0.02

# Input
INPUT_STEP = 0.1

"""
    Game + Hoverboard
"""

class Hoverboard:
    def __init__(self, x = 0.5, y = 0.5, w = 100, h = 10, angle = 0, velocity = None, ang_velocity = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle
        rad = math.pi*self.angle/180
        self.normal = (-math.sin(rad), -math.cos(rad))
        self.velocity = velocity if (velocity != None) else [0,0]
        self.ang_velocity = ang_velocity

        self.last_thrust = [0.5,0.5]
        self.thrust = [0.5,0.5]

        self.surface = pygame.Surface((w,h*5), pygame.SRCALPHA)

    def render(self, screen):
        # clear
        self.surface.fill((0,0,0,0))
        # draw platform
        pygame.draw.rect(self.surface, COLOR_PLAYER, (0, self.h*2, self.w, self.h))
        # draw thruster left
        l = (self.thrust[0]+self.last_thrust[0])/2.0     # smooth display
        th = 2*self.h*l
        for y in range(int(th)):
            for x in range(self.h):
                self.surface.set_at((x,self.h*3+y),
                                (COLOR_THRUSTER_LEFT[0],
                                 COLOR_THRUSTER_LEFT[1],
                                 COLOR_THRUSTER_LEFT[2],
                                 int(255*(1-y/th))))
        # draw thruster right
        r = (self.thrust[1]+self.last_thrust[1])/2.0 # smooth display
        th = 2*self.h*r
        for y in range(int(th)):
            for x in range(self.w-self.h,self.w):
                self.surface.set_at((x,self.h*3+y),
                                (COLOR_THRUSTER_RIGHT[0],
                                 COLOR_THRUSTER_RIGHT[1],
                                 COLOR_THRUSTER_RIGHT[2],
                                 int(255*(1-y/th))))

        # rotate and draw
        rotated = pygame.transform.rotate(self.surface, self.angle)
        rect = rotated.get_rect()
        rect[0] -= rect[2]/2
        rect[1] -= rect[3]/2
        rect[0] += self.x*DISPLAY[0]
        rect[1] += self.y*DISPLAY[1]
        screen.blit(rotated, rect)

    def update(self, delta_t):
        #self.angle += left
        # gravity
        self.velocity[1] += GRAVITY*delta_t
        # thrust
        ang_accel_l = (FORCE/INTERTIA_MOMENTUM)*self.thrust[0]
        ang_accel_r = (FORCE/INTERTIA_MOMENTUM)*self.thrust[1]
        accel = (FORCE/MASS)*(self.thrust[0]+self.thrust[1])

        self.ang_velocity += (ang_accel_r-ang_accel_l)*delta_t

        rad = math.pi*self.angle/180
        self.normal = (-math.sin(rad), -math.cos(rad))
        self.velocity[0] += self.normal[0]*accel*delta_t
        self.velocity[1] += self.normal[1]*accel*delta_t

        # update position and angle
        self.x += self.velocity[0]*delta_t
        self.y += self.velocity[1]*delta_t
        self.angle += self.ang_velocity*delta_t

    def set_thrust(self, left, right):
        self.last_thrust = list(self.thrust)
        self.thrust[0] = left
        if (self.thrust[0] < 0): self.thrust[0] = 0
        elif (self.thrust[0] > 1): self.thrust[0] = 1
        self.thrust[1] = right
        if (self.thrust[1] < 0): self.thrust[1] = 0
        elif (self.thrust[1] > 1): self.thrust[1] = 1

class Game:
    def __init__(self, start_angle = 0, ui = True, network_ui = None):
        pygame.init()
        self.start_angle = start_angle
        self.ui = ui
        self.network_ui = network_ui
        self.reset_flag = False
        self.hoverboard = Hoverboard(angle = start_angle)
        if (self.ui):
            self.screen = pygame.display.set_mode(DISPLAY)

    def render(self):
        self.screen.fill(COLOR_BACKGROUND)
        self.hoverboard.render(self.screen)

        # Thrusters UI
        l = (self.hoverboard.thrust[0]+self.hoverboard.last_thrust[0])/2.0 # smooth display
        r = (self.hoverboard.thrust[1]+self.hoverboard.last_thrust[1])/2.0 # smooth display
        pygame.draw.rect(self.screen, COLOR_THRUSTER_OFF, (THRUSTER[2], (DISPLAY[1]-THRUSTER[1])/2,THRUSTER[0],THRUSTER[1]*(1-l)))
        pygame.draw.rect(self.screen, COLOR_THRUSTER_LEFT, (THRUSTER[2], (DISPLAY[1]-THRUSTER[1])/2+THRUSTER[1]*(1-l),THRUSTER[0],THRUSTER[1]*l))
        pygame.draw.rect(self.screen, COLOR_THRUSTER_OFF, (DISPLAY[0]-THRUSTER[0]-THRUSTER[2], (DISPLAY[1]-THRUSTER[1])/2,THRUSTER[0],THRUSTER[1]*(1-r)))
        pygame.draw.rect(self.screen, COLOR_THRUSTER_RIGHT, (DISPLAY[0]-THRUSTER[0]-THRUSTER[2], (DISPLAY[1]-THRUSTER[1])/2+THRUSTER[1]*(1-r),THRUSTER[0],THRUSTER[1]*r))

        # Network UI
        if (self.network_ui != None):
            self.network_ui.render(self.screen)

    def loop(self):
        last_t = pygame.time.get_ticks()
        while True:
            t = pygame.time.get_ticks()
            # Events
            for event in pygame.event.get():
                # Input
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

    def update(self, delta_t):
        # Clear reset flag
        if (self.reset_flag): self.reset_flag = False
        # Update hoverboard
        self.hoverboard.update(delta_t)
        # Collision (end game)
        if (self.hoverboard.x <= 0 or self.hoverboard.x >= 1 or
            self.hoverboard.y <= 0 or self.hoverboard.y >= 1):
            self.reset()
        # Render
        if (self.ui):
            self.render()
            pygame.display.update()

    def reset(self):
        self.hoverboard = Hoverboard(angle = self.start_angle)
        # Set reset flag
        self.reset_flag = True

if __name__ == "__main__":
   Game().loop()
