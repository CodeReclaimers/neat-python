import numpy as np

def autocorr(sequence):
    """
    Calculate auto-correlation for a given sequence by way of convolution (FFT). High auto-correlation after N
     time shifts implies periodicity in the sequence where N is the period.
    Parameters:
        sequence (numpy array): the sequence to auto-correlate

    returns:
        r (float): a value that express the degree of auto-correlation
        lag (int): the period after which the signal resembles itself the most
    """

    n = sequence.size
    sequence = (sequence - np.mean(sequence)) # normalize the sequence
    result = np.correlate(sequence, sequence, mode='same')
    acorr = result[n//2 + 1:] / (sequence.var() * np.arange(n-1, n//2, -1))
    lag = acorr.argmax() + 1
    r = acorr[lag-1]
    '''
    if np.abs(r) > 0.5:
      print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    else:
      print('Appears to be not autocorrelated')
    '''
    return r, lag

def discrete_differential(sequence):
    differential_sequence = np.empty(len(sequence))
    for i in range(len(sequence)-1):
        differential_sequence[i] = sequence[i+1] - sequence[i]

    differential_sequence[-1] = 0

    return differential_sequence

def find_extrema(differential_sequence):
    extrema_indeces = []
    for i in range(len(differential_sequence)-1):
        if differential_sequence[i] * differential_sequence[i+1] < 0 or differential_sequence[i] == 0:
            extrema_indeces.append(i)

    return extrema_indeces

def is_oscillating(sequence):

    extrema = find_extrema(discrete_differential(sequence))
    corr_coeffficient, period = autocorr(sequence)
    oscillator = True

    if len(extrema) <= 1:
        oscillator = False
    else:
        for i in range(len(extrema)-2):
            if extrema[i+1] == extrema[i]+1: # there should not be extrema at neighbouring indeces
                oscillator = False
                break

    if corr_coeffficient < 0.5: periodic = False
    else: periodic = True


    return corr_coeffficient, period, oscillator and periodic

def freq_analysis(time_series, sampling_freq):
    assert isinstance(time_series, np.ndarray)
    N = time_series.size
    fourier_coeffs = np.fft.fft(time_series)
    freqs = np.fft.fftfreq(N)*sampling_freq

    return fourier_coeffs, freqs

