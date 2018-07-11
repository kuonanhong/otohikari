'''
This scripts reads a video file and extracts the signals
from LEDs according to parameters from a JSON file provided
as argument.

2018 Horiike, modified by Scheibler
'''
from readers import video_stream, frame_grabber, BoxCatcher

import matplotlib.pyplot as plt
import json

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Extract the LED intensity from the video.')
    parser.add_argument('video_file', metavar='FILE', help='The video file')
    parser.add_argument('frame', metavar='N', type=int, help='The frame number')
    parser.add_argument('-s', '--save', type=str, help='Save the values to a file')
    args = parser.parse_args()

    the_frame = frame_grabber(args.video_file, frame=args.frame, show=True)

    if args.save is not None:
        plt.savefig(args.save)
