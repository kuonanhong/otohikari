'''
This scripts reads a video file and extracts the signals
from LEDs according to parameters from a JSON file provided
as argument.

2018 Horiike, modified by Scheibler
'''
import cv2
import sys
import numpy as np
from readers import video_stream, frame_grabber, BoxCatcher

import matplotlib.pyplot as plt
import json

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Extract the LED intensity from the video.')
    parser.add_argument('parameter_file', metavar='FILE', help='The file that contains segmentation information')
    parser.add_argument('-s', '--save', type=str, help='Save the values to a file')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot the calibration curves')
    parser.add_argument('-i', '--inspect', type=float, help='Displays a single frame from the video, then exits')
    args = parser.parse_args()

    with open(args.parameter_file, "r") as f:
        data = json.load(f)

    if args.inspect is not None:
        the_frame = frame_grabber(data['white']['video_file'], frame=args.inspect, show=True)

    else:
        for color, param in data.items():
            print(color,'...')
            img = BoxCatcher([param['loc'][::-1]],[param['h'], param['w']], monitor=True)
            video_stream(param["video_file"],
                    start=param["start"], end=param["end"],
                    callback=img)
            data[color]['data'] = img.extract().tolist()

        if args.save is not None:
            with open(args.save, 'w') as f:
                json.dump(data, f, indent=1)

        if args.plot:

            for color, param in data.items():
                curve = np.array(param['data'])
                t = np.arange(curve.shape[0]) / param['fps']
                if 'color_num' in data[color]:
                    plt.plot(t, curve[...,param['color_num']], label=color, c=color)
                else:
                    plt.plot(t, curve.mean(axis=-1), label=color, c='k')

            plt.title('Calibration curves')
            plt.xlabel('time [sec]')
            plt.ylabel('power')
            plt.legend()
            plt.xlim(0, t[-1])
            plt.show()
