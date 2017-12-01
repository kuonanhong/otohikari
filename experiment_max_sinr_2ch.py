'''
This file applies a max SINR approach using the VAD information
from the LED and the two channels from the camera

Author: Robin Scheibler
Created: 2017/12/01
'''
import argparse, os
import numpy as np
from scipy.io import wavfile
import pyroomacoustics as pra

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('pastel')

from utils import compute_variances, compute_gain

experiment_folder = '../measurements/20171128/segmented'
file_pattern = os.path.join(experiment_folder, '{}_SIR_{}_dB.wav')

sir_choices = [5, 10, 15, 20, 25]
mic_choices = {'camera' : 'camera_audio', 'pyramic' : 'pyramic_audio'}

parser = argparse.ArgumentParser()
parser.add_argument('SIR', type=int, choices=sir_choices, help='The SIR between speech and noise')
parser.add_argument('--thresh', '-t', type=float, help='The threshold for VAD')
parser.add_argument('--nfft', type=int, default=1024, help='The FFT size to use for STFT')
parser.add_argument('--mic', '-m', type=str, choices=mic_choices.keys(), help='Which input device to use')
args = parser.parse_args()

SIR = args.SIR
nfft = args.nfft
vad_thresh = args.thresh

fs_led, leds   = wavfile.read(file_pattern.format('camera_leds', SIR))
fs_snd, audio  = wavfile.read(file_pattern.format(mic_choices[args.mic], SIR))
n_samples = audio.shape[0]  # shorthand
n_channels = audio.shape[1]

# perform VAD
vad = leds > vad_thresh

# now, upsample VAD to audio rate. We handle here
# the fractional rate difference crudely
def fractional_repeat(x, r, max_len=None):

    if max_len is None:
        max_len = int(r * x.shape[0])

    if max_len > np.ceil(r * x.shape[0]):
        raise ValueError('x too short to satisfy max_len')

    y = np.empty(max_len, dtype=x.dtype)

    length = 0
    i = 0
    sample_error = 0
    while length < max_len:

        end = min(max_len, length + int(t_led))
        L = end - length

        y[length:end] = vad[i]

        length = end

        # finish early if necessary
        if end == max_len:
            break

        # keep track of fractonal sample error accumulation
        sample_error += t_led - L

        # adjust when necessary
        if sample_error > 1.:
            y[length] = vad[i]
            sample_error -= 1
            length += 1

        i += 1

    return y

t_led = fs_snd / fs_led
vad_snd = fractional_repeat(vad, t_led, max_len=n_samples)

##############################
## STFT and frame-level VAD ##
##############################

# Now compute the STFT of the microphone input
engine = None
engine = pra.realtime.STFT(nfft, nfft/2, channels=n_channels, analysis_window=pra.hann(nfft))
X = engine.analysis_multiple(audio)
X_time = np.arange(1, X.shape[0]+1) * (nfft / 2) / fs_snd

# This is a wasteful trick to estimate how much VAD in each frame
engine2 = None
engine2 = pra.realtime.STFT(nfft, nfft/2, analysis_window=np.ones(nfft))
vad_stft = engine2.analysis_multiple(vad_snd.astype(np.float))
vad_frac = (np.sum(np.abs(vad_stft)**2, axis=1) * 2 - np.abs(vad_stft[:,0])**2) / nfft ** 2
vad_frames = vad_frac > 0.5  # re-quantize to 0/1

##########################
## MAX SINR BEAMFORMING ##
##########################

VAD = vad_frames
VAD_N = not vad_frames

# covariance matrices
Rs = np.einsum('i...j,i...k->...jk', X[VAD,:,:], np.conj(X[VAD,:,:])) / np.sum(VAD)
Rn = np.einsum('i...j,i...k->...jk', X[VAD_N,:,:], np.conj(X[VAD_N,:,:])) / np.sum(VAD_N)

# compute the MaxSINR beamformer
w = [la.eigh(rs, b=rn, eigvals=(M-1,M-1))[1] for rs,rn in zip(Rs[1:], Rn[1:])]
w = np.squeeze(np.array(w))
w /= la.norm(w, axis=1)[:,None]
w = np.concatenate([np.ones((1,M))/np.sqrt(M), w], axis=0)

# normalize with respect to input signal
z = compute_gain(w, X[vad_x,:,:], ref, n_lambda=None)

##########
## PLOT ##
##########

# time axis for plotting
led_time = np.arange(leds.shape[0]) / fs_led + 1 / (2 * fs_led)
audio_time = np.arange(n_samples) / fs_snd

plt.figure()
plt.plot(led_time, leds, 'r')
plt.title('LED signal')

# match the scales of VAD and light to sound before plotting
q_vad = np.max(audio)
q_led = np.max(audio) / np.max(leds)

plt.figure()
plt.plot(audio_time, audio, 'b') 
plt.plot(led_time, leds * q_led, 'r')
plt.plot(audio_time, vad_snd * q_vad, 'g')
plt.legend(['audio','VAD'])
plt.title('LED and audio signals')
plt.show()
