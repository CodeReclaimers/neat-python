"""A NEAT (NeuroEvolution of Augmenting Topologies) implementation"""

__version__ = '1.1.0'

import neat.nn as nn
import neat.ctrnn as ctrnn
import neat.iznn as iznn

from neat.config import Config
from neat.population import Population, CompleteExtinctionException
from neat.genome import DefaultGenome
from neat.reproduction import DefaultReproduction
from neat.stagnation import DefaultStagnation
from neat.reporting import StdOutReporter
from neat.species import DefaultSpeciesSet
from neat.statistics import StatisticsReporter
from neat.parallel import ParallelEvaluator
from neat.checkpoint import Checkpointer
from neat.innovation import InnovationTracker
from neat.genes import DefaultNodeGene, DefaultConnectionGene
