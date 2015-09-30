""" 2-input XOR example using Izhikevich's spiking neuron model. """
import math
import os

from neat import config, population, chromosome, genome, iznn, visualize

# XOR-2
INPUTS = ((0, 0), (0, 1), (1, 0), (1, 1))
OUTPUTS = (0, 1, 1, 0)

# Maximum amount of simulated time (in milliseconds) we will
# wait for the network to produce an output.
MAX_TIME = 100


def eval_fitness(chromosomes):
    for chromo in chromosomes:
        net = iznn.create_phenotype(chromo)
        error = 0.0
        for inputData, outputData in zip(INPUTS, OUTPUTS):
            for j in range(MAX_TIME):
                output = net.advance([x * 10 for x in inputData])
                if output != [False, False]:
                    break
            if output[0] and not output[1]:  # Network answered 1
                error += (1 - outputData) ** 2
            elif not output[0] and output[1]:  # Network answered 0
                error += (0 - outputData) ** 2
            else:
                # No answer or ambiguous
                error += 1
        chromo.fitness = 1 - math.sqrt(error / len(OUTPUTS))
        if not chromo.fitness:
            chromo.fitness = 0.00001


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config.load(os.path.join(local_dir, 'xor2_config'))

    # Temporary workaround
    chromosome.node_gene_type = genome.NodeGene

    # For spiking networks
    config.Config.output_nodes = 2

    pop = population.Population()
    pop.epoch(eval_fitness, 500, report=True, save_best=False)

    winner = pop.stats()[0][-1]
    print 'Number of evaluations: %d' % winner.id

    # Verify network output against training data.
    print '\nBest network output:'
    net = iznn.create_phenotype(winner)
    for inputData, outputData in zip(INPUTS, OUTPUTS):
        for j in range(MAX_TIME):
            output = net.advance([x * 10 for x in inputData])
            if output != [False, False]:
                break
        if output[0] and not output[1]:  # Network answered 1
            print "%r expected %d got 1" % (inputData, outputData)
        elif not output[0] and output[1]:  # Network answered 0
            print "%r expected %d got 0" % (inputData, outputData)
        else:
            print "%r expected %d got ?" % (inputData, outputData)

    # Visualize the winner network and plot statistics.
    visualize.plot_stats(pop.stats())
    visualize.plot_species(pop.species_log())
    visualize.draw_net(winner, view=True)


if __name__ == '__main__':
    run()
