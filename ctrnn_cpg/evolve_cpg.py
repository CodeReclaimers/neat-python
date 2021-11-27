"""
Evolve a CTRNN with a CPG-like behaviour.
1) Rhythmic/oscillatory output from high level input signal
2) Capable adaptation of output waveform. (Neuromodulation)
"""

import multiprocessing
import os
import pickle
import numpy as np
import matplotlib.pylab as plt

import neat
from ctrnn_cpg import visualize
from ctrnn_cpg import utils

control_signals = np.arange(0.0, 110.0, 10.0)
test_control_signals = np.arange(25, 125, 25)
time_constants = np.arange(0.01, 0.55, 0.05)
run_number = 0
fitness_function = 'rmse'

def simulate(ctrnn, control_sig, simulation_timestep=0.05, simulation_seconds=25.6, variate_input=False):
    """
    Run a CTRNN for a given amount of time steps to retrieve a sequence of output values.
    Parameters:
        ctrnn (CTRNN): an instance of the CTRNN class
        control_sig (float): a value in [0.0, 1.0]

        simulation_timestep (float): the length of each time step in the simulation
        simulation_seconds (float): the number of seconds for the simulation run
        variate_input (boolean): a boolaen to control whether or not to vary the control signal
            during the simulation
    Returns:
        (1-D numpy array): a sequence of the CTRNN's output values over time.
    """
    time = 0.0
    total_timesteps = int(simulation_seconds/simulation_timestep)
    inputs = [control_sig, 0.0]
    out_sequence = []


    while time < simulation_seconds:

        if variate_input:
            timestep_index = np.floor(time * (total_timesteps/simulation_seconds)) #convert seconds into time indeces
            new_signal_index = int(np.floor(timestep_index/(total_timesteps/test_control_signals.size)))
            inputs[0] = test_control_signals[new_signal_index]

        output = ctrnn.advance(inputs, simulation_timestep, simulation_timestep)
        out_sequence.append(output)
        time += simulation_timestep

    return np.asarray(out_sequence).flatten()

def frequency_error(sequence, control_signal):
    """
    Perform a DFT of a CTRNN's output sequence to find the dominant frequency component, then
    calculate a score how close the frequency is to pi*control_singal.

    Parameters:
            sequence (1-D numpy array): a CTRNN's output values over time
            control_signal (float): a value in [0.0, 1.0]
    Returns:
        (float): score calculation in (0.0, 100.0]
    """
    n = sequence.size
    fourier = np.fft.fft(sequence)[:(n//2)+1]
    freqs = np.fft.fftfreq(n)[:(n//2)+1]*20

    max_coeff = np.argmax(np.abs(fourier))
    freq = freqs[max_coeff]
    target_freq = control_signal/10  #scale down control signal into values between [0, 10]

    return np.abs(freq - target_freq)

def exponential_fitness(errors):
    fitness = np.where(errors < 0.01, 10.0, 1/errors)
    return np.prod(fitness)

def rmse_fitness(errors):
    return np.sqrt(np.mean(errors**2))

def eval_genome(genome, config):
    """
    Test and evaluate the fitness of a single genome.
    Rewards for:
        - periodicity by performing auto-correlation on output
        - control of output frequency by adjusting input
        - modulating waveform with input from reflexes.

    Parameters:
        genome (DefaultGenome): the genome for a CTRNN
        config (Config): the configuration of the genomes

    Returns:
        fitness (float): the calculated fitness of the CTRNN
    """
    net = neat.ctrnn.CTRNN.create(genome, config, time_constant=0.1)

    #np.random.shuffle(control_signals)
    sim_results = []

    for c in range(control_signals.size):
        sim_results.append([control_signals[c], simulate(net, control_signals[c])])

    errors = []
    rmse_worst = np.sqrt(np.mean(np.full(10, 20)**2))
    #fitness = rmse_worst
    #print(fitness, fitness_function)
    for result in sim_results:
        r_coefficient, period, oscillating = utils.is_oscillating(result[1])
        if oscillating:
            errors.append(frequency_error(result[1], result[0]))
        else:
            errors.append(rmse_worst)

    if fitness_function == 'rmse':
        fitness = rmse_fitness(np.asarray(errors))
    elif fitness_function == 'exponential':
        fitness = exponential_fitness(np.asarray(errors))

    return fitness

#def eval_genomes(genomes, config):

def run(run_id):
    """
    Run NEAT evolution.
    Parameters: None
    Returns: None
    """
    run_number = run_id

    # Load config file. Assumed to live in the same folder as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ctrnn')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Set up initial population and reporter objects.
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    # Run fitness evaluations in parallell.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate, n=1)

    # Save the winner.
    with open('./ctrnn_cpg/results/config_G/winner-ctrnn{0}'.format(run_number), 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    # Plot output from winner network
    sim_result = simulate(neat.ctrnn.CTRNN.create(winner, config, time_constant=0.1), 0.0, variate_input=True)
    t = np.linspace(0, 25.6, 512)
    plt.figure(figsize=(5, 4))
    plt.plot(t, sim_result)
    plt.xlabel('Time (t)')
    plt.ylabel('Output')
    plt.title('Genome fitness = {:.2f}'.format(winner.fitness))
    plt.vlines(6.4, -2, 2, linestyles='dashed', label='c=50')
    plt.vlines(12.8, -2, 2, linestyles='dashed', label='c=75')
    plt.vlines(19.2, -2, 2, linestyles='dashed', label='c=100')
    plt.savefig('./ctrnn_cpg/results/config_G/test{0}.png'.format(run_number), dpi=300)
    plt.close()

    # Visualizations of evolution and winner network topography.
    #visualize.plot_stats(stats, ylog=True, view=False, filename="./ctrnn_cpg/results/config_G/ctrnn-fitness{0}.svg".format(run_number))
    #visualize.plot_species(stats, view=True, filename="ctrnn-speciation.svg")

    #node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    #visualize.draw_net(config, winner, True, node_names=node_names)

    #visualize.draw_net(config, winner, view=False, node_names=node_names,
    #                  filename="./ctrnn_cpg/results/config_G/winner-ctrnn{0}.gv".format(run_number))
    #visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                   filename="winner-ctrnn-enabled.gv", show_disabled=False)
    #visualize.draw_net(config, winner, view=True, node_names=node_names,
    #                   filename="winner-ctrnn-enabled-pruned.gv", show_disabled=False, prune_unused=True)


