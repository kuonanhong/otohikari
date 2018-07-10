import argparse, json, os
import jsongzip
import numpy as np
import cv2
import chainer
import matplotlib.pyplot as plt
from scipy.io import wavfile
from readers import ThreadedVideoStream, ProcessorBase, PixelCatcher, BoxCatcher
from ml_localization import get_data, models, get_formatters

if __name__ == '__main__':

    video_choices = [ 'noise', 'speech', 'hori_1', 'hori_2', 'hori_3', 'hori_4', 'hori_5', ]
    path_choices = ['diagonal', 'parallel_long', 'parallel_short']

    parser = argparse.ArgumentParser(description='Show the estimated source location on top of the video')
    parser.add_argument('protocol', type=str,
            help='The protocol file containing the experiment metadata')
    parser.add_argument('config', type=str, help='The JSON file containing the configuration.')
    parser.add_argument('-v', '--video', type=str, choices=video_choices,
            default=video_choices[0], help='The video segment to process')
    parser.add_argument('-p', '--path', type=str, choices=path_choices,
            default=path_choices[0], help='The path to process')
    args = parser.parse_args()

    # get the path to the experiment files
    experiment_path = os.path.split(args.protocol)[0]

    # Read some parameters from the protocol
    with open(args.protocol,'r') as f:
        protocol = json.load(f)

    # read in the important stuff
    blinkies = np.array(protocol['blinky_locations'])
    video_path = os.path.join(experiment_path, protocol['videos'][args.video])

    # read_in blinky data
    blinky_fn = os.path.join(experiment_path,
            'processed/{}_blinky_signal.wav'.format(args.video))
    _, blinky_sig = wavfile.read(blinky_fn)

    # read in the groundtruth locations
    source_loc_fn = os.path.join(experiment_path,
            'processed/{}_source_locations.json.gz'.format(args.video))
    if os.path.exists(source_loc_fn):
        source_locations = jsongzip.load(source_loc_fn)
    else:
        source_locations = None

    # Get the Neural Network Model
    with open(args.config, 'r') as f:
        config = json.load(f)

    # import model and use MSE
    nn = models[config['model']['name']](*config['model']['args'], **config['model']['kwargs'])
    chainer.serializers.load_npz(config['model']['file'], nn)

    # Create some random colors (BGR)
    color_gt = [34, 139, 34]
    color_est = [51, 51, 255]

    # find start of segment
    if args.video in protocol['video_segmentation']:
        f_start, f_end = protocol['video_segmentation'][args.video][args.path]
        f_start = int(f_start * protocol['video_info']['fps'])
        f_end = int(f_end * protocol['video_info']['fps'])
    else:
        f_start, f_end = 0, None

    i_frame = f_start
    nf = 2

    if source_locations is not None:
        while source_locations[0][0] < i_frame:
            source_locations.pop(0)

    with ThreadedVideoStream(video_path, start=f_start, end=f_end) as cap:

        cap.start()

        while cap.is_streaming():

            frame = cap.read()

            if frame is None:
                break

            if i_frame >= nf:
                # perform localization
                nn_in = np.mean(blinky_sig[i_frame-nf:i_frame+nf+1,:], axis=0)
                #nn_in -= nn_in.min()
                nn_in /= nn_in.max()
                nn_in = nn_in[None,:].astype(np.float32)

                y, x = nn(nn_in).data[0,:]
                frame = cv2.circle(frame, (x,y), 5, color_est, -1)

                # check if groundtruth is available for that location
                if source_locations is not None and i_frame == source_locations[0][0]:
                    _, [y, x] = source_locations.pop(0)
                    frame = cv2.circle(frame, (x,y), 5, color_gt, -1)

                cv2.imshow('frame', frame)

            i_frame += 1

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    cv2.destroyAllWindows()
