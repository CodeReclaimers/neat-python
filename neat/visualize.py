# -*- coding: UTF-8 -*-
import graphviz
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(best_genomes, avg_scores, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """

    generation = range(len(best_genomes))

    fitness = [c.fitness for c in best_genomes]

    plt.plot(generation, avg_scores, 'b-', label="average")
    plt.plot(generation, fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None):
    """ Plots the trains for a single spiking neuron. """
    plt.title("Izhikevich's spiking neuron model")
    plt.ylabel("Membrane Potential")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(spikes, "g-")
    if filename is not None:
        plt.savefig(filename)
    if view:
        plt.show()
    plt.close()


def plot_species(species_log, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    num_generations = len(species_log)
    print num_generations
    num_species = max(map(len, species_log))
    curves = []
    for gen in species_log:
        species = [0] * num_species
        species[:len(gen)] = gen
        curves.append(np.array(species))
    curves = np.array(curves).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(chromosome, view=False, filename=None):
    """ Receives a chromosome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    # Attributes for network input nodes.
    input_attrs = {
        'style': 'filled',
        'shape': 'box'}

    # Attributes for network output nodes.
    output_attrs = {
        'style': 'filled',
        'color': 'lightblue'}

    # Attributes for disabled network connections.
    disabled_attrs = {
        'style': 'dotted',
        'color': 'cornflowerblue'}

    dot = graphviz.Digraph(format='svg', node_attr=node_attrs)

    for ng_id, ng in chromosome.node_genes.items():
        if ng.type == 'INPUT':
            dot.node(str(ng_id), _attributes=input_attrs)

    for ng_id, ng in chromosome.node_genes.items():
        if ng.type == 'OUTPUT':
            dot.node(str(ng_id), _attributes=output_attrs)

    for cg in chromosome.conn_genes.values():
        a = str(cg.in_node_id)
        b = str(cg.out_node_id)
        if cg.enabled is False:
            dot.edge(a, b, _attributes=disabled_attrs)
        else:
            dot.edge(a, b)

    dot.render(filename, view=view)
