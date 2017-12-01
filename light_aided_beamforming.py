'''
This is a simple example of beamforming where the VAD function is provided by a
sound-to-light sensor
'''

import numpy as np
from scipy.io import wavfile
from scipy import signal as sig
from scipy import linalg as la
import pyroomacoustics as pra
import matplotlib.pyplot as plt

from utils import compute_variances, compute_gain
from light_array import LightArray2

### Parameters ###
fs_sound = 16000
fs_light = 30
nfft = 512  # approximately 30 overlapping frames / second

vad_thresh = -30  # dB

r, target_audio = wavfile.read('../audio/DR1_FAKS0_SX133_SX313_SX223.wav')
assert r == fs_sound
target_audio = np.concatenate([np.zeros(fs_sound), target_audio.astype(np.float32)])
target_audio /= target_audio.max()
sigma_s = np.std(target_audio)
sound_time = np.arange(target_audio.shape[0]) / fs_sound

mics_loc = np.c_[
        [1, 1.9825],  # mic 1
        [1, 2.0175],  # mic 2
        ]  # location of microphone array
mics_loc = pra.circular_2D_array([2., 2.], 6, 0, 0.0625)  # kindof compactsix
src_loc = np.r_[5, 2]  # source location
noise_loc = np.r_[2.5, 4.5]

SIR = 15  # decibels
SINR = SIR - 1  # decibels

sigma_i, sigma_n = compute_variances(SIR, SINR, src_loc, noise_loc, mics_loc.mean(axis=1), sigma_s=sigma_s)

interference_audio = np.random.randn(target_audio.shape[0] + fs_sound) * sigma_i

room = pra.ShoeBox([6,5], fs=16000, max_order=12, absorption=0.4, sigma2_awgn=sigma_n**2)
room.add_source(src_loc, signal=target_audio)
room.add_source(noise_loc, signal=interference_audio)

# conventional microphone array
M = mics_loc.shape[1]
mics = pra.Beamformer(mics_loc, fs=fs_sound, N=nfft, hop=nfft / 2, zpb=nfft)
room.add_microphone_array(mics)

room.simulate()

# sound-to-light sensor
# we assume there is no propagation delay between speaker and sensor
leds = LightArray2(src_loc, fs=fs_light)
leds.record(target_audio + np.random.randn(*target_audio.shape) * sigma_n, fs=fs_sound)
leds_sig = leds.signals - leds.signals.min()
leds_sig /= leds_sig.max()
leds_time = np.arange(leds.signals.shape[0]) / fs_light

# perform VAD on the light signal
vad = leds.signals > vad_thresh

# Now compute the STFT of the microphone input
engine = None
engine = pra.realtime.STFT(nfft, nfft/2, channels=mics.M, analysis_window=pra.hann(nfft))
X = engine.analysis_multiple(room.mic_array.signals.T)
X_time = np.arange(1, X.shape[0]+1) * (nfft / 2) / fs_sound

# we need to match the VAD to sampling rate of X
vad_x = np.zeros(X_time.shape[0], dtype=bool)
v = 0
for i,t in enumerate(X_time):
    if v < leds_time.shape[0] - 1 and  abs(t - leds_time[v]) > abs(t - leds_time[v+1]):
        v += 1
    vad_x[i] = vad[v]
vad_x_comp = np.logical_not(vad_x)

# covariance matrix
Rs = np.einsum('i...j,i...k->...jk', X[vad_x,:,:], np.conj(X[vad_x,:,:])) / np.sum(vad_x)
Rn = np.einsum('i...j,i...k->...jk', X[vad_x_comp,:,:], np.conj(X[vad_x_comp,:,:])) / np.sum(vad_x_comp)

# compute the MaxSINR beamformer
w = [la.eigh(rs, b=rn, eigvals=(M-1,M-1))[1] for rs,rn in zip(Rs[1:], Rn[1:])]
w = np.squeeze(np.array(w))
w /= la.norm(w, axis=1)[:,None]
w = np.concatenate([np.ones((1,M))/np.sqrt(M), w], axis=0)

# Compute the gain
ref = X[vad_x,:,0]

#z = compute_gain(w, X[vad_x,:,:], ref, n_lambda=20, clip_up=1.0, clip_down=0.1)
#z = compute_gain(w, X[vad_x,:,:], ref, n_lambda=20)
#z = compute_gain(w, X[vad_x,:,:], ref, n_lambda=None, clip_up=2.0)
z = compute_gain(w, X[vad_x,:,:], ref, n_lambda=None)
#z = compute_gain(w, X[vad_x,:,:], ref, n_lambda=20)

sig_in = pra.normalize(mics.signals[0])

mics.weights = w.T

room.plot(img_order=1, freq=[800,1000,1200, 1400, 1600, 1800, 2000])
plt.figure()
mics.plot_beam_response()

sig_out_flat = mics.process()
sig_out_flat = pra.normalize(sig_out_flat)

mics.weights = (z[:,None] * w).T
sig_out_ref0 = mics.process()
sig_out_ref0 = pra.normalize(sig_out_ref0)

room.plot(img_order=1, freq=[800,1000,1200, 1400, 1600, 1800, 2000])

plt.figure()
mics.plot_beam_response()

plt.figure()
plt.plot(sound_time, target_audio)
plt.plot(X_time, vad_x)
plt.plot(leds_time, leds_sig, '--')
plt.legend(['sound','VAD', 'light'])

plt.figure()
mics.plot()

plt.show()
