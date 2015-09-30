# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------#
# A parallel version of XOR using the nice Parallel Python module       #
#                http://www.parallelpython.com/                         #
#                                                                       #
# Since XOR is a simple experiment, a parallel version won't actually   #
# take any advantages of it due to overhead and transfer-communication. #
# The example below is only a general idea of how to implement a        #
# parallel experiment in neat-python.                                   #
# ----------------------------------------------------------------------#
import math
import os
import time

import pp

from neat import config, population, chromosome, genome, visualize
from neat.nn import nn_pure as nn

t0 = time.time()

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config.load(os.path.join(local_dir, 'xor2_config'))

# Temporary workaround
chromosome.node_gene_type = genome.NodeGene

# Create Parallel Python server
secret = 'my_secret_key'
job_server = pp.Server(ncpus=2, secret=secret)
print "Starting pp with", job_server.get_ncpus(), "workers"


def eval_fitness(population):
    # number of chunks (jobs)
    num_chunks = job_server.get_ncpus()
    # size for each chunk
    size = config.Config.pop_size/num_chunks
    # make sure we have a proper number of chunks
    assert config.Config.pop_size % num_chunks == 0, "Population size is not multiple of num_chunks"

    jobs = []
    for k in xrange(num_chunks):
        # divide the population in chunks and evaluate each chunk on a
        # different processor or machine
        # print 'Chunk %d:  [%3d:%3d]' %(k, size*k, size*(k+1))
        jobs.append(job_server.submit(parallel_evaluation,
                                      args=(population[size*k:size*(k+1)], ),
                                      depfuncs=(),
                                      modules=('neat','math')))

    all_jobs =[] # the results for all jobs
    for k in xrange(num_chunks):
        all_jobs += (jobs[k]())
    # assign the fitness back to each chromosome
    for i, fitness in enumerate(all_jobs):
        population[i].fitness = fitness


def parallel_evaluation(chunk):
    # This function will run in parallel
    from neat.nn import nn_pure as nn

    # XOR-2
    INPUTS = ((0, 0), (0, 1), (1, 0), (1, 1))
    OUTPUTS = (0, 1, 1, 0)

    fitness = []
    for c in chunk:
        net = nn.create_ffphenotype(c)

        error = 0.0
        for i, input in enumerate(INPUTS):
            output = net.sactivate(input) # serial activation
            error += (output[0] - OUTPUTS[i])**2

        fitness.append(1 - math.sqrt(error/len(OUTPUTS)))

    # when finished, return the list of fitness values
    return fitness


pop = population.Population()
pop.epoch(eval_fitness, 400, report=False)

print "total evolution time %.3f sec" % (time.time() - t0)
print "time per generation %.3f sec" % ((time.time() - t0) / pop.generation)

job_server.print_stats()

winner = pop.stats()[0][-1]
print 'Number of evaluations: %d' % winner.id

# Visualize the winner network and plot statistics.
visualize.draw_net(winner)
visualize.plot_stats(pop.stats())
visualize.plot_species(pop.species_log())

# Let's check if it's really solved the problem
print '\nBest network output:'
net = nn.create_ffphenotype(winner)
INPUTS = ((0, 0), (0, 1), (1, 0), (1, 1))
OUTPUTS = (0, 1, 1, 0)
for i, inputs in enumerate(INPUTS):
    output = net.sactivate(inputs)  # serial activation
    print "%1.5f \t %1.5f" % (OUTPUTS[i], output[0])

