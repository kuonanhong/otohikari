import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra

from light_array import LightArray


fs_snd, speech = wavfile.read('../audio/fq_sample1.wav')

fs_lit = 30

# create anechoic room
field = pra.ShoeBox([100, 100], fs_snd, absorption=1., max_order=0)

# microphone
mic = LightArray([[1.,2.],[1.,1.]], fs_lit)
field.add_microphone_array(mic)

# sources
field.add_source([5., 1.], signal=speech)

# run simulation
field.simulate()

xcorr = pra.correlate(mic.signals[0], mic.signals[1], interp=20)
plt.plot(xcorr)

print(pra.tdoa(*(mic.signals), interp=int(4 * fs_snd / fs_lit), fs=fs_lit, phat=False) * pra.constants.get('c'))
#print(pra.tdoa(mic.signals[0], mic.signals[1], interp=1, fs=fs_snd, phat=True) * pra.constants.get('c'))

plt.figure()
plt.plot(field.rir[0][0])
plt.plot(field.rir[1][0])
