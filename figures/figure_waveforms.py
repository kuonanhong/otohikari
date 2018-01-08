'''
Create a figure that compares the signal at the microphone to the LED signal
exctracted from the video stream.
'''
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile

experiment_folder = '../measurements/20171207'
file_pattern = os.path.join(experiment_folder, 'segmented/{}_{}_SIR_{}_dB.wav')
sir_choices = [5, 10, 15, 20, 25]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('SIR', nargs='?', default=25, type=int, choices=sir_choices, help='The SIR between speech and noise')
    args = parser.parse_args()

    SIR = args.SIR

    # read_in the mix signals
    fs_led, leds   = wavfile.read(file_pattern.format('camera_leds_zero_hold', 'mix', SIR))
    fs_snd, audio  = wavfile.read(file_pattern.format('camera_audio', 'mix', SIR))
    assert len(leds) == len(audio)
    assert fs_led == fs_snd

    leds = leds.astype(np.float)
    leds -= np.min(leds)
    leds *= np.max(audio[:,0]) / np.max(leds)

    fs_slow, leds_s = wavfile.read(file_pattern.format('camera_leds', 'mix', SIR))
    leds_s = leds_s.astype(np.float)
    leds_s -= leds_s[0]
    leds_s *= np.max(audio[:,0]) / np.max(leds_s)
    leds_s -= 500
    leds_s *= np.max(audio[:,0]) / np.max(leds_s)

    time = np.arange(len(audio)) / fs_snd - (1 / fs_slow / 2)
    time_s = np.arange(len(leds_s)) / fs_slow

    # Plot
    sns.set(style='white', context='paper', font_scale=0.9,
            rc={
                'axes.facecolor': (0, 0, 0, 0),
                'figure.figsize':(3.38649, 3.338649 * 0.25),
                'lines.linewidth':0.25,
                'text.usetex': False,
                })
    sns.set_palette(sns.color_palette("cubehelix", 3))

    plt.plot(time, audio[:,0])
    plt.plot(time_s[0:-2], leds_s[2:], linewidth=1.2)
    plt.legend(['audio', 'blinky'], fontsize='x-small')
    plt.xticks([])
    plt.yticks([])
    sns.despine(left=True, bottom=True)
    plt.tight_layout(pad=0.02)

    plt.savefig('figures/waveforms_example.pdf')

    plt.show()
