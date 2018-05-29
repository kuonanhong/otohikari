import matplotlib
matplotlib.use('Qt5Agg')

import argparse, os, json
from scipy.io import wavfile
import numpy as np

import matplotlib.pyplot as plt

experiment_folder = '../measurements/20180523'
session = 'session-20180523/'
metadata_file = os.path.join(experiment_folder, session + 'processed/metadata.json')
protocol_file = os.path.join(experiment_folder, session + 'protocol.json')

with open(metadata_file, 'r') as f:
    metadata = json.load(f)

with open(protocol_file, 'r') as f:
    protocol = json.load(f)

file_pattern = os.path.join(experiment_folder, metadata['filename_pattern'])

sir_choices = [-5, 0, 5, 10, 15, 20]
mix_str = 'mix'
source_str = lambda ch : 'ch' + str(ch+1)
sources = ['mix'] + [source_str(i) for i in range(4)]
fs = 16000

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('SIR', nargs='?', type=int, choices=sir_choices,
            help='The SIR between speech and noise')
    args = parser.parse_args()

    plt.figure()
    N = len(sources)
    for n, source in enumerate(sources):

        _, blinky = wavfile.read(file_pattern.format(
            mic='blinky_red', snr=args.SIR, source=source, fs=fs))

        _, camera = wavfile.read(file_pattern.format(
            mic='camera', snr=args.SIR, source=source, fs=fs))

        _, pyramic = wavfile.read(file_pattern.format(
            mic='pyramic', snr=args.SIR, source=source, fs=fs))

        if source == 'mix':
            scale = np.maximum(camera.max(), pyramic.max())

        assert blinky.shape[0] == camera.shape[0]
        assert camera.shape[0] == pyramic.shape[0]

        time_vec = np.arange(blinky.shape[0]) / fs

        plt.subplot(N, 1, n + 1)

        if source == 'mix':
            b_sig = blinky[:,n]
        else:
            b_sig = blinky[:,int(source[-1]) - 1]

        plt.plot(time_vec, b_sig, 'r')
        plt.plot(time_vec, pyramic[:,0] / scale, 'b')
        plt.plot(time_vec, camera[:,0] / scale, 'g')
        plt.xlim([time_vec[0], time_vec[-1]])

        plt.legend(['blinky', 'pyramic', 'camera'])

    plt.show()






