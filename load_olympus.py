
import argparse
import os
import platform
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp

from utilities import OnlineStats, ffmpeg_open_raw_video, ffmpeg_open_raw_audio

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Name of the video file')

args = parser.parse_args()

# video information
framesize_v = (1280, 720)
n_channels_v = 3
fs_v = 30.0003
v_bufsize = 3 * int(fs_v)  # collect 1s (approx) of video frames
v_readsize = 2 * int(fs_v)

# audio information
fs_a = 44100
n_channels_a = 2
dtype_a = np.int16
a_bufsize = 5 * fs_a  # 5 sec of audio
a_readsize = fs_a


# compute some statistics on the input feed

stats = OnlineStats((framesize_v[1], framesize_v[0], n_channels_v))

video = ffmpeg_open_raw_video(args.filename, framesize_v, n_channels=n_channels_v, 
                            bufsize=v_bufsize, readsize=v_readsize, online_func=stats.process)

audio = ffmpeg_open_raw_audio(args.filename, n_channels=n_channels_a, 
                            bufsize=a_bufsize, readsize=a_readsize)

# find the pixel with most variance
y, x, ch = np.unravel_index(np.argmax(stats.var), stats.var.shape)

# plot the profiles
def display_light_sound(x, y):

    from cycler import cycler
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    # now plot
    light_signal = video[:,y,x,:]
    plt.subplot(2,1,1)
    light_time = np.arange(light_signal.shape[0]) / fs_v
    plt.plot(light_time, light_signal)

    plt.subplot(2,1,2)
    audio_time = np.arange(audio.shape[0]) / fs_a
    plt.plot(audio_time, audio)

    plt.show()

display_light_sound(x, y)


