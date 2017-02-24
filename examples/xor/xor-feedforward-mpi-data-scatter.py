"""
Implementation of neat feedforward XOR example with basic mpi4py.

Motivation for MPI implementation come from bottle neck of sharing
large training data sets between process/compute nodes.

To run XOR example, neat population.py requires patch that initialized
two additional variables, neat.population.run is note used at all.

Tested on Cluster with Debian Jessie 8.7, Python 2.7, MPICH2 v3.1, mpi4py v2.0

Linux Commands to install mpich and mpi4py:
    sudo apt-get install mpich libmpich-dev
    sudo apt-get install python-dev
    wget https://bootstrap.pypa.io/get-pip.py
    sudo python get-pip.py
    sudo pip install --upgrade pip
    sudo apt-get install python-pip
    sudo pip install numpy mpi4py

Parallel Model:

              / activate genomes on data slice 1 \
generation 0 -- activate genomes on data slice 2 --> evaluate, reproduce, broadcast neat -> generation N
              \ activate genomes on data slice n /


Training data structure:
    Training data is an nested array that consists of Sessions Arrays and Each Session
    is a collection of Frames Arrays. Each Frame Array in Frames is divided into neat input/output.


Command to run example on single node with 5 processes:
"mpiexec -n 5 /usr/bin/python evolve-feedforward-mpi-data-scatter.py"

"""
from __future__ import print_function
import os
from mpi4py import MPI
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import SafeConfigParser as ConfigParser

# Add path to neat-python to import neat if neat is not installed on a system
import sys
sys.path.append("/path/to/neat-python")

import neat
from neat.six_util import iteritems, itervalues

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()

if rank == 0:
    print ("Number of processes", comm_size)


def slicer(session_length, step, _size, buffer_size, _id):
    """Slices Each Session into frames frames indexes"""
    _Indexes = range(0, session_length)
    start_Indexes = _Indexes[::step]
    end_Indexes = _Indexes[_size::step]
    Indexes = zip(start_Indexes, end_Indexes)
    if rank == 0:
        print ("Total Number of Frames", len(Indexes))
    buffers_Indexes = map(None, *(iter(Indexes),) * buffer_size)
    return [buffers_Indexes, _id]

_neat = []
buffers_indexes = []

# Initialize neat class, generate training data indexes and broadcast
if rank == 0:
    local_dir = os.path.dirname('__file__')
    config_file = os.path.join(local_dir, "config-xor-feedforward-mpi-data-scatter")

    # Initialize population.py with minor changes
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    _neat = neat.Population(config)
    _neat.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    _neat.add_reporter(stats)
    print (rank, "loading data")

    # load parameters
    # Can be integrated into config.py so user can pass own parameters
    with open(config_file) as fp:
        parameters = ConfigParser()
        parameters.readfp(fp)
        if not parameters.has_section('UserData'):
            raise RuntimeError("'UserData' section not found in NEAT configuration file.")
        frame_size = parameters.getint('UserData', 'frame_size')
        frame_step = parameters.getint('UserData', 'frame_step')
        # self.max_generations, added to population.py
        _neat.max_generations = parameters.getint('NEAT', 'max_generations')

    # Data Example with 1 and 2 Session

    # sessions = [
    #     [[0, 0, 0,
    #       0, 1, 1,
    #       1, 0, 1,
    #       1, 1, 0], 0],
    #
    #     [[0, 0, 1,
    #       0, 1, 0,
    #       1, 0, 0,
    #       1, 1, 1], 1]]

    sessions = [
        [[0, 0, 0,
          0, 1, 1,
          1, 0, 1,
          1, 1, 0], 0]]

    # generate frames indexes for each session
    for frames in sessions:
        length = len(frames[0]) + 1
        buffers_indexes.append(slicer(length, frame_step, frame_size, comm_size, frames[1]))

# Broadcast indexies to all nodes
buffers_indexes = comm.bcast(buffers_indexes, root=0)

# Load chuck of training data on each node
training_data = []
for sessions_indexes in buffers_indexes:
    session_id = sessions_indexes[1]
    for indexes in sessions_indexes[0]:
        _buffer = []
        if rank == 0:
            frames = sessions[session_id][0]
            for index in indexes:
                if index:
                    _buffer.append([frames[index[0]:index[1]],
                                    index[0], index[1], session_id])
                else:
                    _buffer.append(None)

        # Rank N, Receives data chunk from rank 0
        training_data_ = comm.scatter(_buffer, root=0)

        # if data exist collect chunk of data on each node in a training_data array
        if training_data_:
            training_data.append(training_data_)

# print("training data on rank", rank, "data", training_data)

if rank == 0:
    print("data is loaded on all nodes")

# Force to write data to terminal
sys.stdout.flush()

# Broadcast neat object to all nodes
_neat = comm.bcast(_neat, root=0)

# Run Generations
for gen in range(_neat.max_generations):

    # Report only from rank 0
    if rank == 0:
        _neat.reporters.start_generation(gen)

    genomes = list(iteritems(_neat.population))

    genomes_outputs_g = []

    # ALL Nodes: evaluate genomes and return network outputs
    # Data frame is split into input frame (data[0:n] and output frame (data[n:])
    for data in training_data:
        if data:
            for genome_id, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, _neat.config)
                output = net.activate(data[0][0:2])
                genomes_outputs_g.append([rank, gen, data,
                                          genome_id, data[0][2], output])

    # TODO change this to comm.Gather ?
    # gather all net outputs on rank 0
    _genomes_outputs = comm.gather(genomes_outputs_g, root=0)

    # To make sure all nodes finished processing data/genomes, Sync nodes
    comm.Barrier()

    # On rank 0, evaluate fitness, reproduce and report statistics.
    if rank == 0:

        genomes_outputs = []
        _best_genome = None

        # TODO Combine gathering outputs and fitness calculation
        # Gather outputs into single list
        for outputs in _genomes_outputs:
            # print("outputs", outputs)
            if outputs:
                genomes_outputs += outputs

        # For each genome and genome_outputs calculate fitness function
        for genome in genomes:
            genome[1].fitness = 4.0
            for output in genomes_outputs:
                if output[3] == genome[0]:
                    genome[1].fitness -= (output[5][0] - output[4]) ** 2

        # Find best genome in current generation and report statistics.
        for g in itervalues(_neat.population):
            if _best_genome is None or g.fitness > _best_genome.fitness:
                _best_genome = g
        _neat.reporters.post_evaluate(_neat.config, _neat.population,
                                      _neat.species, _best_genome)

        # Track the best genome ever seen.
        if _neat.best_genome is None or _best_genome.fitness > _neat.best_genome.fitness:
            _neat.best_genome = _best_genome

        # Check if fitness threshold is reached.
        # TODO fv can be calculated when fitness function is calculated
        fv = _neat.fitness_criterion(g.fitness for g in itervalues(_neat.population))
        if fv >= _neat.config.fitness_threshold:
            _neat.reporters.found_solution(_neat.config, _neat.generation, _best_genome)
            _neat.break_generation = True

        else:
            # Generate next generation from the current generation.
            _neat.population = _neat.reproduction.reproduce(_neat.config, _neat.species,
                                                            _neat.config.pop_size, gen)
            # Check for complete extinction.
            if not _neat.species.species:
                _neat.reporters.complete_extinction()

                if _neat.config.reset_on_extinction:
                    _neat.population = _neat.reproduction.create_new(_neat.config.genome_type,
                                                                     _neat.config.genome_config,
                                                                     _neat.config.pop_size)

                    _neat.species.speciate(_neat.config, _neat.population, gen)
                    _neat.reporters.end_generation(_neat.config, _neat.population, _neat.species)

                else:
                    _neat.break_generation = "Extinction"

            else:
                # Divide the new population into species.
                _neat.species.speciate(_neat.config, _neat.population, gen)
                _neat.reporters.end_generation(_neat.config, _neat.population, _neat.species)

        sys.stdout.flush()

    _neat = comm.bcast(_neat, root=0)

    # self.break_generation added to population.py
    # Break if solution fund or population extinct,
    # to avoid deadlock break must be executed on all nodes,
    if _neat.break_generation:
        break

if rank == 0 and _neat.break_generation != "Extinction":
    print("\nBest Genome Input/Output")
    net = neat.nn.FeedForwardNetwork.create(_neat.best_genome, _neat.config)
    for sessions_indexes in buffers_indexes:
        session_id = sessions_indexes[1]
        for indexes in sessions_indexes[0]:
            frames = sessions[session_id][0]
            for index in indexes:
                if index:
                    net_input = frames[index[0]:index[1]]
                    output = net.activate(net_input[0:2])
                    print("input {!r}, expected output {!r}, got {!r}".format(
                        net_input[0:2], net_input[2], output[0]))
    print("\nBest Genome Network")
    print(_neat.best_genome)
