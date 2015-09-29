# -*- coding: UTF-8 -*-
import random

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

try:
    import biggles

    def plot_stats(stats, ylog=False):
        """ Plots the population's average and best fitness. """
        generation = [i for i in xrange(len(stats[0]))]

        fitness = [c.fitness for c in stats[0]]
        avg_pop = [avg for avg in stats[1]]

        plot = biggles.FramedPlot()
        plot.title = "Population's average and best fitness"
        plot.xlabel = r"Generations"
        plot.ylabel = r"Fitness"

        plot.add(biggles.Curve(generation, fitness, color="red"))
        plot.add(biggles.Curve(generation, avg_pop, color="blue"))
        plot.ylog = ylog

        # plot.show() # X11
        plot.write_img(600, 300, 'avg_fitness.svg')
        # width and height doesn't seem to affect the output!


    def plot_spikes(spikes):
        """ Plots the trains for a single spiking neuron. """
        time = [i for i in xrange(len(spikes))]

        plot = biggles.FramedPlot()
        plot.title = "Izhikevich's spiking neuron model"
        plot.ylabel = r"Membrane Potential"
        plot.xlabel = r"Time (in ms)"

        plot.add(biggles.Curve(time, spikes, color="green"))
        plot.write_img(600, 300, 'spiking_neuron.svg')
        # width and height doesn't seem to affect the output!


    def plot_species(species_log):
        """ Visualizes speciation throughout evolution. """
        plot = biggles.FramedPlot()
        plot.title = "Speciation"
        plot.ylabel = r"Size per Species"
        plot.xlabel = r"Generations"
        generation = [i for i in xrange(len(species_log))]

        species = []
        curves = []

        for gen in xrange(len(generation)):
            for j in xrange(len(species_log), 0, -1):
                try:
                    species.append(species_log[-j][gen] + sum(species_log[-j][:gen]))
                except IndexError:
                    species.append(sum(species_log[-j][:gen]))
            curves.append(species)
            species = []

        s1 = biggles.Curve(generation, curves[0])

        plot.add(s1)
        plot.add(biggles.FillBetween(generation, [0] * len(generation), generation, curves[0],
                                     color=random.randint(0, 90000)))

        for i in range(1, len(curves)):
            c = biggles.Curve(generation, curves[i])
            plot.add(c)
            plot.add(
                biggles.FillBetween(generation, curves[i - 1], generation, curves[i], color=random.randint(0, 90000)))

        plot.write_img(1024, 800, 'speciation.svg')

except ImportError:
    print "The python2-biggles library is not installed. If you wish to plot some nice statistics, " + \
          ", please install it: https://pypi.python.org/pypi/python2-biggles"

    def plot_stats(stats, ylog):
        print 'The python2-biggles library is not installed, unable to plot statistics.'

    def plot_spikes(spikes):
        print 'The python2-biggles library is not installed, unable to plot statistics.'

    def plot_species(species_log):
        print 'The python2-biggles library is not installed, unable to plot statistics.'


try:
    import graphviz

    def draw_net(chromosome, filename=None):
        """ Receives a chromosome and draws a neural network with arbitrary topology. """
        dot = graphviz.Digraph(format='svg', node_attr=node_attrs)

        for ng in chromosome.node_genes:
            if ng.type == 'INPUT':
                dot.node(str(ng.id), _attributes=input_attrs)

        for ng in chromosome.node_genes:
            if ng.type == 'OUTPUT':
                dot.node(str(ng.id), _attributes=output_attrs)

        for cg in chromosome.conn_genes:
            a = str(cg.innodeid)
            b = str(cg.outnodeid)
            if cg.enabled is False:
                dot.edge(a, b, _attributes=disabled_attrs)
            else:
                dot.edge(a, b)

        dot.render(filename, view=True)

except ImportError:
    print "The Python graphviz library is not installed. If you wish to generate a graphical representation " + \
          "of the resulting neural network, please install it: https://pypi.python.org/pypi/graphviz"

    def draw_net(chromosome, filename=None):
        print 'The graphviz library is not installed, unable to render network.'




