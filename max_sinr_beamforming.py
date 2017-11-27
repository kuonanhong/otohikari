import numpy as np
import matplotlib.pyplot as plt

def compute_variances(SIR, SINR, source_loc, interference_loc, mic_loc, sigma_s=1):
    '''
    This function will compute the powers (i.e. variance) of an interference
    source and the microphone self noise power given values for the
    SIR (signal-to-interference ratio) and SINR (signal-to-interference-and-noise ratio)
    and optionaly a target source power.
    '''

    if SINR > SIR:
        raise ValueError('SINR must be less or equal to SIR.')

    d_s = np.linalg.norm(source_loc - mic_loc)
    d_i = np.linalg.norm(interference_loc - mic_loc)

    sigma_i = sigma_s * d_i / d_s * 10**(-SIR / 20)
    sigma_n = sigma_s / d_s * np.sqrt(10**(-SINR / 10) - 10**(-SIR / 10))

    return sigma_i, sigma_n

def compute_gain(w, X, ref, n_lambda=20, clip_up=None, clip_down=None):
    '''

    Parameters
    ----------
    w: array_like (n_bins, n_channels)
    X: array_like (n_frames, n_bins, n_channels)
        The STFT data
    ref: array_like (n_frames, n_bins)
        The reference signal
    n_lambda: int, optional
        The number of lagrange multiplier value to try in the approximation (default: 20)
    clip_up: float, optional
        Limits the maximum value of the gain (default no limit)
    clip_down: float, optional
        Limits the minimum value of the gain (default no limit)
    '''


    if n_lambda is None:
        lag = np.zeros(1)
    else:
        lag = np.logspace(-10, 5, n_lambda)

    let_weights = [np.zeros(n_lambda)]

    left_hand = np.sum(np.sum(X, axis=0) * np.conj(w), axis=1)
    right_hand = np.sum(ref, axis=0)

    left_sqm = np.abs(left_hand)**2

    G = (left_sqm[:,None] / (left_sqm[:,None] + lag[None,:])) * right_hand[:,None]
    c_candidates = G / left_hand[:,None]

    if n_lambda is not None:
        weights = np.linalg.lstsq(G, right_hand)[0]
        c = np.squeeze(np.dot(c_candidates, weights))
    else:
        c = c_candidates[:,0]

    if clip_up is not None:
        I = np.abs(c) > clip_up
        c[I] *= clip_up / np.abs(c[I])

    if clip_down is not None:
        I = np.abs(c) < clip_down
        c[I] *= clip_down / np.abs(c[I])

    return np.conj(c)

