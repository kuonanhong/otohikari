
import argparse
import numpy as np

from utilities import OnlineStats, ffmpeg_open_raw_video, ffmpeg_open_raw_audio
from check_calibration import check_calibration

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

stats = OnlineStats((framesize_v[1], framesize_v[0], n_channels_v))

video = ffmpeg_open_raw_video(args.filename, framesize_v, n_channels=n_channels_v, 
                            bufsize=v_bufsize, readsize=v_readsize, online_func=stats.process)
video_time = np.arange(video.shape[0]) / fs_v

audio = ffmpeg_open_raw_audio(args.filename, n_channels=n_channels_a, dtype=dtype_a,
                            bufsize=a_bufsize, readsize=a_readsize)
audio_time = np.arange(audio.shape[0]) / fs_a

check_calibration(video, video_time, audio, audio_time, stats)
