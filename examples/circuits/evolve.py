import os
from random import choice, random

import numpy as np
import matplotlib.pyplot as plt

from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit

import neat
from neat.activations import ActivationFunctionSet
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import BaseGene
from neat.six_util import iteritems, iterkeys

import visualize

class CircuitNodeGene(BaseGene):
    __gene_attributes__ = []

    def distance(self, other, config):
        return 0.0


class CircuitConnectionGene(BaseGene):
    __gene_attributes__ = [StringAttribute('component'),
                           FloatAttribute('value'),
                           BoolAttribute('enabled')]

    def distance(self, other, config):
        d = abs(self.value - other.value)
        if self.component != other.component:
            d += 1.0
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient


class CircuitGenomeConfig(object):
    __params = [ConfigParameter('num_inputs', int),
                ConfigParameter('num_outputs', int),
                ConfigParameter('compatibility_disjoint_coefficient', float),
                ConfigParameter('compatibility_weight_coefficient', float),
                ConfigParameter('conn_add_prob', float),
                ConfigParameter('conn_delete_prob', float),
                ConfigParameter('node_add_prob', float),
                ConfigParameter('node_delete_prob', float)]

    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        self.activation_options = params.get('activation_options', 'sigmoid').strip().split()
        self.aggregation_options = params.get('aggregation_options', 'sum').strip().split()

        # Gather configuration data from the gene classes.
        self.__params += CircuitNodeGene.get_config_params()
        self.__params += CircuitConnectionGene.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self.__params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

    def save(self, f):
        write_pretty_params(f, self, self.__params)


class CircuitGenome(object):
    @classmethod
    def parse_config(cls, param_dict):
        return CircuitGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}

        # Fitness results.
        self.fitness = None

    def mutate(self, config):
        """ Mutates this genome. """

        # TODO: Make a configuration item to choose whether or not multiple
        # mutations can happen simultaneously.
        if random() < config.node_add_prob:
            self.mutate_add_node(config)

        if random() < config.node_delete_prob:
            self.mutate_delete_node(config)

        if random() < config.conn_add_prob:
            self.mutate_add_connection(config)

        if random() < config.conn_delete_prob:
            self.mutate_delete_connection()

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in iteritems(parent1.connections):
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in iteritems(parent1_set):
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

    def get_new_node_key(self):
        new_id = 0
        while new_id in self.nodes:
            new_id += 1
        return new_id

    def mutate_add_node(self, config):
        if not self.connections:
            return None, None

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = self.get_new_node_key()
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id)
        self.add_connection(config, new_node_id, o)

    def add_connection(self, config, input_key, output_key):
        # TODO: Add validation of this connection addition.
        key = (input_key, output_key)
        connection = CircuitConnectionGene(key)
        connection.init_attributes(config)
        self.connections[key] = connection

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        possible_outputs = list(iterkeys(self.nodes))
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)

        if in_node == out_node:
            return

        # # Don't duplicate connections.
        # key = (in_node, out_node)
        # if key in self.connections:
        #     return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    def mutate_delete_node(self, config):
        # Do nothing if there are no non-output nodes.
        available_nodes = [(k, v) for k, v in iteritems(self.nodes) if k not in config.output_keys]
        if not available_nodes:
            return -1

        del_key, del_node = choice(available_nodes)

        connections_to_delete = set()
        for k, v in iteritems(self.connections):
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        return del_key

    def mutate_delete_connection(self):
        if self.connections:
            key = choice(list(self.connections.keys()))
            del self.connections[key]

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in iterkeys(other.nodes):
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in iteritems(self.nodes):
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance + config.compatibility_disjoint_coefficient * disjoint_nodes) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in iterkeys(other.connections):
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in iteritems(self.connections):
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance + config.compatibility_disjoint_coefficient * disjoint_connections) / max_conn

        distance = node_distance + connection_distance

        return distance

    def size(self):
        """Returns genome 'complexity', taken to be (number of nodes, number of enabled connections)"""
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled is True])
        return len(self.nodes), num_enabled_connections

    def __str__(self):
        s = "Nodes:"
        for k, ng in iteritems(self.nodes):
            s += "\n\t{0} {1!s}".format(k, ng)
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s

    def add_hidden_nodes(self, config):
        for i in range(config.num_hidden):
            node_key = self.get_new_node_key()
            assert node_key not in self.nodes
            node = self.__class__.create_node(config, node_key)
            self.nodes[node_key] = node

    def configure_new(self, config):
        # Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        for input_id in config.input_keys:
            for node_id in iterkeys(self.nodes):
                connection = self.create_connection(config, input_id, node_id)
                self.connections[connection.key] = connection

    @staticmethod
    def create_node(config, node_id):
        node = CircuitNodeGene(node_id)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = CircuitConnectionGene((input_id, output_id))
        connection.init_attributes(config)
        return connection


def get_pins(key):
    pins = []
    for k in key:
        if k < 0:
            pins.append('input{0}'.format(-k))
        else:
            pins.append('node{0}'.format(k))

    return pins


def create_circuit(genome, config):
    libraries_path = '/home/alan/ngspice/libraries'  # os.path.join(os.path.dirname(os.path.dirname(__file__)), 'libraries')
    spice_library = SpiceLibrary(libraries_path)

    circuit = Circuit('NEAT')
    circuit.include(spice_library['1N4148'])

    Vbase = circuit.V('base', 'input1', circuit.gnd, 2)
    Vcc = circuit.V('cc', 'input2', circuit.gnd, 5)
    Vgnd = circuit.V('gnd', 'input3', circuit.gnd, 0)
    #circuit.R('test1', 'node0', circuit.gnd, 1e6)
    #circuit.R('test2', 'node0', 'input1', 1e6)
    ridx = 1
    xidx = 1
    for key, c in iteritems(genome.connections):
        if c.component == 'resistor':
            pin0, pin1 = get_pins(key)
            R = 10 ** c.value
            circuit.R(ridx, pin1, pin0, R)
            ridx += 1
        elif c.component == 'diode':
            pin0, pin1 = get_pins(key)
            circuit.X(xidx, '1N4148', pin1, pin0)
            xidx += 1

    return circuit

V_MAX = 2.0

def get_expected(inputs):
    #return 0.5 * inputs - 0.15 * inputs ** 2
    return 2.0 / (1.0 + np.exp(-2.5 * (inputs - 1.0)))


def simulate(genome, config):
    try:
        circuit = create_circuit(genome, config)
        #print(str(circuit))

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.dc(Vbase=slice(0, V_MAX, .01))

        inputs = np.array(analysis.input1)
        outputs = np.array(analysis.node0)
        expected = get_expected(inputs)

        return -np.sqrt(np.sum((expected - outputs) ** 2) / len(outputs))
    except Exception as e:
        return -10.0


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = simulate(genome, config)


def run(config_file):
    # Load configuration.
    config = neat.Config(CircuitGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    #config.save('test_save_config.txt')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 1000 generations.
    pe = neat.ParallelEvaluator(4, simulate)
    p.run(pe.evaluate, 1000)

    # Write run statistics to file.
    stats.save()

    # Display the winning genome.
    winner = stats.best_genome()
    print('\nBest genome:\nfitness {!s}\n{!s}'.format(winner.fitness, winner))

    winner_circuit = create_circuit(winner, config)
    print(winner_circuit)

    simulator = winner_circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.dc(Vbase=slice(0, V_MAX, .01))

    inputs = np.array(analysis.input1)
    outputs = np.array(analysis.node0)
    expected = get_expected(inputs)

    plt.plot(inputs, outputs, 'r-', label='output')
    plt.plot(inputs, expected, 'g-.', label='target')
    plt.grid()
    plt.legend(loc='best')
    plt.gca().set_aspect(1)
    plt.show()

    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=False)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    print(local_dir, os.getcwd())
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
