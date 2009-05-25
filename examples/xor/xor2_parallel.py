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
import random
import zlib
import cPickle as pickle
from neat import config, population, chromosome, genome, visualize

#random.seed(5465)
config.load('xor2_config')

config.Config.max_fitness_threshold = 0.9
config.Config.pop_size = 150
# Temporary workaround
chromosome.node_gene_type = genome.NodeGene

import pp
# list of available servers
#servers = ("server1:port1, "server2:port2", "server3:port3")
servers = () # empty: run local
# you can set the number of CPUs the local machine will use
# if you set ncpus = 0, then only distributed workers will
# receive and do the job
job_server = pp.Server(ncpus=1, ppservers=servers, loglevel=20)
print "Starting pp with", job_server.get_ncpus(), "workers"

def eval_fitness(population):

    # number of chunks (jobs)
    num_chunks = 2
    # size for each subpopulation
    size = config.Config.pop_size/num_chunks
    # make sure we have a proper number of chunks
    assert config.Config.pop_size % num_chunks == 0, "Population size is not multiple of num_chunks"

    jobs = []
    for k in xrange(num_chunks):
        # divide the population in chunks and evaluate each chunk on a
        # different processor or machine
        print 'Chunk %d:  [%3d:%3d]' %(k, size*k, size*(k+1))

        # compressing the population is useful when running
        # on a network of machines over the internet
        # It drastically reduces the lag at minimal cost
        # zlib compression (for small chromosomes)
        #           ratio             ratio
        # level   protocol=0 (note that protocol 1 or 2 are even better)
        #   1       74.06%
        #   3       77.21%
        #   6       79.68%
        #   9       80.58%

        # first pickles the population object
        pickle_pop = pickle.dumps(population[size*k:size*(k+1)], 2)
        # then compress the pickled object
        compressed_pop = zlib.compress(pickle_pop, 3)
        #print "Ratio: ", len(pickle_pop), len(compressed_pop)
        # submit the job
        jobs.append(job_server.submit(parallel_evaluation,
                                      args=(compressed_pop, k),
                                      depfuncs=(),
                                      modules=('neat','zlib','math')))

    all_jobs =[] # the results for all jobs
    for k in xrange(num_chunks):
        all_jobs += (jobs[k]())
    # assign the fitness back to each chromosome
    for i, fitness in enumerate(all_jobs):
        population[i].fitness = fitness

def parallel_evaluation(compressed_pop, chunk):
    # This function will run in parallel
    from neat.nn import nn_pure as nn

    # don't print OS calls to stdout:
    #http://www.parallelpython.com/component/option,com_smf/Itemid,29/topic,103.0
    print "Evaluating chunk %d at %s" %(chunk, os.popen("hostname").read())

    # decompress the pickled object
    decompress_pop = zlib.decompress(compressed_pop)
    # unpickle it
    sub_pop = pickle.loads(decompress_pop)

    # XOR-2
    INPUTS = ((0, 0), (0, 1), (1, 0), (1, 1))
    OUTPUTS = (0, 1, 1, 0)

    fitness = []
    for c in sub_pop:
        net = nn.create_ffphenotype(c)

        error = 0.0
        for i, input in enumerate(INPUTS):
            output = net.sactivate(input) # serial activation
            error += (output[0] - OUTPUTS[i])**2

        fitness.append(1 - math.sqrt(error/len(OUTPUTS)))

    # when finished, return the list of fitness values
    return fitness


population.Population.evaluate = eval_fitness

pop = population.Population()
pop.epoch(400, report=1, save_best=False)
#visualize.draw_ff(pop.stats[0][-1])
job_server.print_stats()
