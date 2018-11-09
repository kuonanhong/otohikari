'''
This scripts reads a video file and extracts the signals
from LEDs according to parameters from a JSON file provided
as argument.

The JSON file should have the following structure

    {
        "locations" : [
            [ y1, x1 ],
            [ y2, x2 ]
            ],
        "fps" : 30,
        "box" : [ h, w ],
        "start" : 0,
        "end" : 350
    }

Parameters
----------
locations: list of pairs of int
    A list of pairs of coordinates [y, x] where y and x are the vertical and
    horizontal pixel indices, respectively
box: pair of int
    A pair of integers (height, width) defining a box around the location that
    is averaged
fps: int, optional
    The frame rate of the video (default, 30)
start: int, optional
    The start frame number
end: int, optional
    The end frame number

2018 Horiike, modified by Scheibler
'''

import sys, os, argparse
import cv2
import numpy as np
from readers import video_stream, frame_grabber, BoxCatcher

import matplotlib.pyplot as plt
import json

if __name__ == '__main__':

    format_choices = ['.json', '.wav']

    parser = argparse.ArgumentParser(description='Extract the LED intensity from the video.')
    parser.add_argument('parameter_file', metavar='FILE', help='The file that contains segmentation information')
    parser.add_argument('video_file', metavar='VIDEO', help='The path to the video file')
    parser.add_argument('-s', '--save', type=str, help='Save the values to a file')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot the calibration curves')
    parser.add_argument('-i', '--inspect', type=float, help='Displays a single frame from the video, then exits')
    args = parser.parse_args()

    if args.save is not None:
        fmt = os.path.splitext(args.save)[1]
        if fmt not in format_choices:
            raise ValueError('Only the following formats are supported: ' + ' '.join(format_choices))

    with open(args.parameter_file, "r") as f:
        config = json.load(f)

    if 'fps' in config:
        fps = config['fps']
    else:
        fps = 30

    if 'start' in config:
        start = config['start']
    else:
        start = 0

    if 'end' in config:
        end = config['end']
    else:
        end = None

    if args.inspect is not None:
        the_frame = frame_grabber(args.video_file, frame=args.inspect, show=True)

    else:
        img = BoxCatcher(config['locations'], config['box'], monitor=True)
        video_stream(args.video_file,
                start=start, end=end,
                callback=img)

        data = img.extract()
        data_col = dict([[col_str, data[:,:,col].tolist()] for col, col_str in enumerate(['red', 'green', 'blue'])])

        if args.save is not None:
            if fmt == '.json':
                with open(args.save, 'w') as f:
                    json.dump(data_col, f, indent=1)
            elif fmt == '.wav':
                from scipy.io import wavfile
                for col, dat in data_col.items():
                    fn, xt = os.path.splitext(args.save)
                    new_fn = fn + '_' + col + xt
                    wavfile.write(new_fn, fps, np.array(dat).astype(np.uint8))

        if args.plot:

            for color, dat in data_col.items():
                curve = np.array(dat)[:,0]
                t = np.arange(curve.shape[0]) / fps
                plt.plot(t, curve, label=color, c=color)

            plt.title('Blinky #0')
            plt.xlabel('time [sec]')
            plt.ylabel('power')
            plt.legend()
            plt.xlim(0, t[-1])
            plt.show()
