# coding: utf-8
import numpy as np
from scipy.signal import argrelmax, argrelmin
from scipy.interpolate import splrep, splev

n_points = 1000
f1, f2 = 5, 10
time_samples = np.linspace(0, 1, n_points)
mode1 = np.sin(2 * np.pi * f1 * time_samples)
mode2 = np.sin(2 * np.pi * f2 * time_samples)
trend = time_samples
noise = np.random.normal(scale=0.25, size=time_samples.shape)
signal = mode1 + mode2 + trend + noise

tol = 0.05


@profile
def ndiff_extrema_zcrossing(x):
    """Get the difference between the number of zero crossings and extrema."""
    n_max = argrelmax(x)[0].shape[0]
    n_min = argrelmin(x)[0].shape[0]
    n_zeros = (x[:-1] * x[1:] < 0).sum()
    return abs((n_max + n_min) - n_zeros)


@profile
def simple_sift(x):
    maxima = argrelmax(x)[0]
    minima = argrelmin(x)[0]
    x_upper = np.zeros((maxima.shape[0] + 2,))
    x_upper[1:-1] = maxima
    x_upper[-1] = x.shape[0] - 1
    x_lower = np.zeros((minima.shape[0] + 2,))
    x_lower[1:-1] = minima
    x_lower[-1] = x.shape[0] - 1
    tck = splrep(x_upper, x[x_upper.astype(int)])
    upper_envelop = splev(np.arange(x.shape[0]), tck)
    tck = splrep(x_lower, x[x_lower.astype(int)])
    lower_envelop = splev(np.arange(x.shape[0]), tck)
    mean_amplitude = np.abs(upper_envelop - lower_envelop) / 2
    local_mean = (upper_envelop + lower_envelop) / 2
    amplitude_error = np.abs(local_mean) / mean_amplitude
    return x - local_mean, amplitude_error.sum()


@profile
def emd(x, n_imfs):
    imfs = np.zeros((n_imfs + 1, x.shape[0]))
    for i in xrange(n_imfs):
        mode = x - imfs.sum(0)
        while (ndiff_extrema_zcrossing(mode) > 1):
            mode, amplitude_error = simple_sift(mode)
            if amplitude_error <= tol:
                break
        imfs[i, :] = mode
    imfs[-1, :] = x - imfs.sum(0)
    return imfs

n_imfs = 3
imfs = emd(signal, n_imfs)
