"""

    neat-python Neural Network GUI

    Small lib for drawing a neat-python network using pygame

    @author: Hugo Aboud (@hugoaboud)

"""

import random
import pygame
import math

# Dimensions and Positions
DISPLAY = [512,512]
NN_PADDING_TOP = 32
NN_INPUT_X = 128
NN_OUTPUT_X = 384
NN_Y_STEP = 25
NN_NODE_RADIUS = 7
NN_CONN_THICK = 1
NN_INPUT_NAMES = ['      velocity X',
                  '      velocity Y',
                  'angular velocity',
                  '        normal X',
                  '        normal Y']
NN_OUTPUT_COLORS = [(200,0,127),
                    (0,127,200)]

# Colors
COLOR_TEXT = (200,200,200)
COLOR_NN_NODE_INPUT = ((200, 0, 0), (0, 0, 200))
COLOR_NN_NODE = {'relu' : ((200, 100, 100), (100, 100, 200)),
                 'tanh' : ((100, 200, 100), (200, 100, 200)),
                 'gauss' : ((100, 100, 200), (200, 100, 100)),
                 'softplus' : ((100, 100, 200), (200, 100, 100)),
                 'sigmoid' : ((100, 100, 200), (200, 100, 100))}
COLOR_NN_CONNECTION = ((200, 100, 100), (100, 100, 200))
NN_WEIGHT_SCALE = 10

##
#   Utility
##

# Interpolate colors using t=[-1,1]
# t = -1 -> colors[0]
# t = 0 -> (0,0,0)
# t = 1 -> colors[1]

def mixcolor(colors, t):
    # clamp t
    if (t < -1): t = -1
    elif (t > 1): t = 1
    # interpolate color
    color = [0,0,0]
    if (t > 0): color = [t*colors[1][0],t*colors[1][1],t*colors[1][2]]
    elif (t < 0): color = [-t*colors[0][0],-t*colors[0][1],-t*colors[0][2]]
    return color

##
#   GUI
##

class NeuralNetworkGUI:

    class Node:
        def __init__(self, x, y, color, inputs):
            self.last_value = 0
            self.value = 0
            self.x = x
            self.y = y
            self.color = color
            self.inputs = inputs

        def render(self, screen):
            pygame.draw.circle(screen, mixcolor(self.color,(self.last_value+self.value)/2.0), (self.x, self.y), NN_NODE_RADIUS)

        def set(self, value):
            self.last_value = self.value
            self.value = value
            if (self.value < 0): self.value = 0
            elif (self.value > 1): self.value = 1

    def __init__(self, generation, genome, species, net, info=True):
        self.generation = generation
        self.genome = genome
        self.species = species
        self.net = net
        self.info = info

        self.nodes = {}
        self.hidden = []
        self.height = NN_PADDING_TOP+(len(self.net.input_nodes)-1)*NN_Y_STEP

        # input nodes
        inputs = self.net.input_nodes
        for i in range(len(inputs)):
            self.nodes[inputs[i]] = self.Node(NN_INPUT_X, NN_PADDING_TOP+i*NN_Y_STEP, COLOR_NN_NODE_INPUT, {})

        # hidden + output nodes
        outputs = self.net.output_nodes
        for id, node in self.genome.nodes.items():
            # create node on a random position
            x = random.randrange(NN_INPUT_X,NN_OUTPUT_X)
            y = random.randrange(NN_PADDING_TOP,NN_PADDING_TOP+max((len(inputs),len(outputs))))
            self.nodes[id] = self.Node(x, y, COLOR_NN_NODE[node.activation], {ids[0]:conn.weight for ids,conn in self.genome.connections.items() if ids[1] == id})
            # output nodes (specific position and color)
            if (id == outputs[0]):
                self.nodes[id].color = ((0,0,0), NN_OUTPUT_COLORS[0])
                self.nodes[id].x = NN_OUTPUT_X
                self.nodes[id].y = NN_PADDING_TOP+1.5*NN_Y_STEP
                self.nodes[id].value = 0.5
            elif (id == outputs[1]):
                self.nodes[id].color = ((0,0,0), NN_OUTPUT_COLORS[1])
                self.nodes[id].x = NN_OUTPUT_X
                self.nodes[id].y = NN_PADDING_TOP+2.5*NN_Y_STEP
                self.nodes[id].value = 0.5
            # hidden nodes (add to list)
            else:
                self.hidden.append(id)

        # Initialize pygame modules
        pygame.init()

        # Font for UI
        self.font = pygame.font.SysFont(None, 12)

    # moves the hidden nodes around to make the graph more readable
    def relax(self, factor=0.01):
        for id in self.hidden:
            hnode = self.nodes[id]
            # get average of distances
            avg = 0
            dists = {}
            for nid, node in self.nodes.items():
                if (id == nid): continue
                dists[nid] = math.sqrt((hnode.x-node.x)**2+(hnode.y-node.y)**2)
                avg += dists[nid]
            avg /= len(self.nodes)-1
            # move trying to keep all distances on the average
            for nid, node in self.nodes.items():
                if (id == nid): continue
                dist = dists[nid]
                if (dist != 0):
                    dir = ((node.x-hnode.x)/dist,(node.y-hnode.y)/dist)
                    hnode.x += dir[0]*(dist-avg)*factor
                    hnode.y += dir[1]*(dist-avg)*factor
            # clamp Y coordinates
            if (hnode.y < NN_PADDING_TOP): hnode.y = NN_PADDING_TOP
            elif (hnode.y > self.height): hnode.y = self.height

    # render the network
    def render_net(self, screen):
        # normalize net values
        values = self.net.values[self.net.active].copy()
        values[-3] /= 360

        # render node inputs
        for id, node in self.nodes.items():
            for id, weight in node.inputs.items():
                b = self.nodes[id]
                pygame.draw.line(screen, mixcolor(COLOR_NN_CONNECTION,b.value*weight/NN_WEIGHT_SCALE), (node.x,node.y),(b.x,b.y), NN_CONN_THICK)

        # render nodes
        for id, node in self.nodes.items():
            if (id in values):
                node.set(values[id])
            node.render(screen)
        # render input names
        for i in range(len(NN_INPUT_NAMES)):
            img = self.font.render(NN_INPUT_NAMES[i], True, COLOR_TEXT)
            screen.blit(img, (NN_INPUT_X-70,NN_PADDING_TOP+i*NN_Y_STEP-6))

    # relax and render
    def render(self, screen):
        # relax and render network graph
        self.relax()
        self.render_net(screen)
        # if info enabled, render info texts
        if (self.info):
            img = self.font.render(str('GENERATION: {0}'.format(self.generation)), True, COLOR_TEXT)
            screen.blit(img, (10,DISPLAY[1]-45))
            img = self.font.render(str('SPECIES: {0}'.format(self.species)), True, COLOR_TEXT)
            screen.blit(img, (10,DISPLAY[1]-30))
            img = self.font.render(str('ID: {0}'.format(self.genome.key)), True, COLOR_TEXT)
            screen.blit(img, (10,DISPLAY[1]-15))
            img = self.font.render(str('FITNESS: {0}'.format(self.genome.fitness)), True, COLOR_TEXT)
            screen.blit(img, (DISPLAY[0]-120,DISPLAY[1]-15))
