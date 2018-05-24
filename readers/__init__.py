'''
Implements some readers for video and audio streams based on opencv and ffmpeg

Code by Robin Scheibler and Daiki Horiike, 2018
'''
from .video import ThreadedVideoStream, video_stream, frame_grabber
from .audio import audioread
