import argparse

import os
import numpy as np
from scipy import signal as sig
from scipy.io import wavfile
import samplerate

from readers import ffmpeg_open_raw_video, ffmpeg_open_raw_audio, OnlineStats, PixelCatcher

parser = argparse.ArgumentParser()
parser.add_argument('video_file', help='Name of the video recording file')
parser.add_argument('audio_file', help='Name of the audio recording file')
parser.add_argument('output_dir', help='The location of the directory where to save the new files')
group = parser.add_mutually_exclusive_group()
group.add_argument('--pixel', '-p', nargs=2, type=int, action='append', help='Pixel location to extract')
group2 = parser.add_mutually_exclusive_group()
group2.add_argument('--resample', '-r', type=int, default=16000)
group2.add_argument('--no_resample', '-nr', action='store_true')


args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

########################
## VIDEO: Audio Track ##
########################

# audio information
fs_a = 44100
n_channels_a = 2
dtype_a = np.int16
a_bufsize = 5 * fs_a  # 5 sec of audio
a_readsize = fs_a

# Get the audio from the camera
audio_camera = ffmpeg_open_raw_audio(args.video_file, n_channels=n_channels_a, dtype=dtype_a,
                            bufsize=a_bufsize, readsize=a_readsize)
audio_time = np.arange(audio_camera.shape[0]) / fs_a

# Now identify a part of the signal where there is active power
duration = 1.  # seconds
interval = int(duration * fs_a)
pwr = np.sum(
        np.sum(audio_camera.astype(np.float)**2, axis=1)
        [:audio_camera.shape[0] - (audio_camera.shape[0] % interval)]
        .reshape(-1,interval), axis=1)
i_pwr = np.argmax(pwr) * interval
start = i_pwr / fs_a

# resample if needed
if not args.no_resample:
    audio_camera = samplerate.resample(audio_camera, args.resample / fs_a, 'sinc_best')
    fs_a = args.resample

new_name = os.path.splitext(os.path.split(args.video_file)[-1])[0]
wavfile.write(os.path.join(args.output_dir, new_name + '_audio_' + str(fs_a // 1000) + 'kHz.wav'), fs_a, audio_camera.astype(np.int16))

########################
## VIDEO: Video Track ##
########################

# video information
framesize_v = (1280, 720)
#framesize_v = (640, 480)
n_channels_v = 3
fs_v = 30
v_bufsize = 3 * int(fs_v)  # collect 1s (approx) of video frames
v_readsize = 2 * int(fs_v)

# Now read only this subpart of the video
# use this to find the location of the "best" pixel
stats = OnlineStats((framesize_v[1], framesize_v[0], n_channels_v))
video = ffmpeg_open_raw_video(args.video_file, framesize_v, n_channels=n_channels_v, 
                            bufsize=v_bufsize, readsize=v_readsize, online_func=stats, 
                            start=start, duration=duration)
video_time = np.arange(video.shape[0]) / fs_v

# The pixel location with the largest variance signal
if args.pixel is None:
    pixel_locs = [tuple(np.unravel_index(np.argmax(np.sum(stats.var, axis=2)), stats.var.shape[:2]))]
else:
    pixel_locs = [tuple(pixel) for pixel in args.pixel]

catcher = PixelCatcher(pixel_locs)
ffmpeg_open_raw_video(args.video_file, framesize_v, n_channels=n_channels_v, 
                            bufsize=v_bufsize, readsize=v_readsize, online_func=catcher, 
                            keep_data=False)

leds = np.sum(catcher.extract(), axis=2)  # sum RGB channels

# save the output to a wav file
new_name = os.path.splitext(os.path.split(args.video_file)[-1])[0]
pixels_str = '_'.join(['p_{}_{}'.format(p[0],p[1]) for p in pixel_locs])
wavfile.write(os.path.join(args.output_dir, new_name + '_pixels_' + pixels_str + '.wav'), fs_v, leds.astype(np.int16))

######################################
## AUDIO: From the Microphone Array ##
######################################

r, maudio = wavfile.read(args.audio_file)

if not args.no_resample:
    maudio = samplerate.resample(maudio, args.resample / r, 'sinc_best')
    r = args.resample

new_name = os.path.splitext(os.path.split(args.audio_file)[-1])[0]
wavfile.write(os.path.join(args.output_dir, new_name + '_' + str(r // 1000) + 'kHz.wav'), r, maudio.astype(np.int16))
