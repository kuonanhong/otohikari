import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra


fs_snd, speech = wavfile.read('../audio/fq_sample1.wav')

fs_lit = 60

# create anechoic room
field = pra.ShoeBox([100, 100], fs_snd, absorption=1., max_order=0)

# microphone
mic = pra.MicrophoneArray(np.array([[1.,2.],[1.,1.]]), fs_snd)
field.add_microphone_array(mic)

# sources
field.add_source([5., 1.], signal=speech)

# run simulation
field.simulate()

# now we do the sound to power transform
block = fs_snd // fs_lit

pwr = (mic.signals) ** 2
pwr_lf = pwr[:,:-(pwr.shape[1] % block)].reshape((pwr.shape[0],-1,block)).mean(axis=2)

xcorr = pra.correlation(pwr_lf[0], pwr_lf[1], interp=20)
plt.plot(xcorr)

print(pra.tdoa(*pwr_lf, interp=4*block, fs=fs_lit, phat=False) * pra.constants.get('c'))
print(pra.tdoa(mic.signals[0], mic.signals[1], interp=1, fs=fs_snd, phat=True) * pra.constants.get('c'))

plt.figure()
plt.plot(field.rir[0][0])
plt.plot(field.rir[1][0])
