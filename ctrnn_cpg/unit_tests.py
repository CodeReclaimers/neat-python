import os
import multiprocessing
import pickle
import numpy as np
import matplotlib.pylab as plt

from ctrnn_cpg import evolve_cpg as cpg
from ctrnn_cpg import utils
import neat

def test_eval_genome():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-test')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)


    pop = neat.Population(config)
    #pop.population.pop(2) #Population size minimum during creation is 2, so we remove one to reduce it to 1.
    #print(pop.population)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), cpg.eval_genome)
    pop.run(pe.evaluate, n=5)

#def test_autocorr():

def test_discrete_differential():
    seq = np.array([0, 2, 2, -2, -2, 2, 2])
    diff_seq = utils.discrete_differential(seq)
    print(diff_seq)
    t = np.linspace(-25, 25, 200)
    sinus = np.sin(2*t)
    diff_sinus = utils.discrete_differential(sinus)

    plt.plot(t, sinus)
    plt.plot(t, diff_sinus)
    plt.xlabel('t')
    plt.ylabel('sin(t)')
    plt.axis('tight')
    plt.show()

def test_find_extrema():
    seq = np.array([0, 2, 2, -2, -2, 2, 2])
    diff_seq = utils.discrete_differential(seq)
    print(diff_seq)
    extrema = utils.find_extrema(diff_seq)
    print(extrema)

    t = np.linspace(-25, 25, 200)
    sinus = np.sin(t)
    diff_sinus = utils.discrete_differential(sinus)
    extrema_sinus = utils.find_extrema(diff_sinus)
    print(extrema_sinus)

def test_autocorr():
    t = np.linspace(-25, 25, 200)
    sinus = np.sin(t)
    r, l = utils.autocorr(sinus)

def test_is_oscillating():
    t = np.linspace(0, 50, 200)
    sinus = np.sin(t)
    line = np.ones(50)
    a = 1/t
    b = np.sqrt(t)
    cosinus = np.cos(t)
    signal = sinus + cosinus
    r_coefficient, period, rhythmic = utils.is_oscillating(signal)
    print(r_coefficient, rhythmic)
    #print(utils.is_oscillating(line))
    #print(utils.is_oscillating(a))
    #print(utils.is_oscillating(b))
    #print(utils.is_oscillating(cosinus))

def test_frequency_error():
    t = np.linspace(0, 25, 50)
    Fs = 50/25
    signal = np.cos(t)
    error = cpg.frequency_error(signal, 1.0)
    print(error)

def test_simulate():
    print("Running test: test_simulate.")
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-test')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    genome = pop.population.pop(1)
    net = neat.ctrnn.CTRNN.create(genome, config, time_constant=0.25)
    sim_result = cpg.simulate(net, 0.0, simulation_timestep=0.01, variate_input=True)

    print(sim_result.size)
    print(sim_result.shape)
    plt.plot(sim_result)
    plt.vlines([100, 200, 300], -1, 1, linestyles='dashed', label='c')
    plt.show()

def test_freq_anlysis():
    t = np.linspace(0, 10, 1000)
    time_series = np.cos(8*t)
    sampling_freq = 1/(10/1000)
    A, F = utils.freq_analysis(time_series, sampling_freq)

    plt.plot(F, A)
    plt.show()
    plt.plot(t, time_series)
    plt.show()

def dummy_eval(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 50.0
        net = neat.ctrnn.CTRNN.create(genome, config, time_constant=0.1)
        if genome_id % 2 == 0:
            genome.fitness -= 5.0
        else:
            genome.fitness -= 10.0

def test_min_pop():
    print("Running test: test_min_pop()")
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-test')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    print(pop.run(dummy_eval, n=2))

if __name__ == '__main__':
    #test_eval_genome()
    #test_discrete_differential()
    #test_find_extrema()
    #test_autocorr()
    #test_is_oscillating()
    #test_score_frequency()
    #test_simulate()
    #test_freq_anlysis()
    test_min_pop()