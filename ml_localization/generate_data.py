'''
This script generates simulated power intensity data to train
a localization network.
'''
import itertools, json, os

import numpy as np
from scipy.io import wavfile

import pyroomacoustics

import ipyparallel as ipp
from joblib import Parallel, delayed
from functools import partial

parameters = dict(
        # simulation parameters
        sample_length = 0.5,      # length of audio sample in seconds
        gain_range = [-3., 3., 5],  # source gains, 5 points from -3dB to 3dB
        grid_step = 0.05,
        n_test = 1000,

        # room parameters
        room_dim = [6,5],
        fs_sound = 16000,
        max_order = 12,
        absorption = 0.4,
        sigma2_awgn = 0.4,

        # light array parameters
        mic_array = [
            [ 1.5, 1.5 ],
            [ 2.5, 1.5 ],
            [ 3.5, 1.5 ],
            [ 4.0, 2.5 ],
            [ 3.5, 3.5 ],
            [ 2.5, 3.5 ],
            [ 1.5, 3.5 ],
            [ 1.0, 2.5 ]
            ],
        fs_light = 32,

        # output parameters
        output_dir = '/data/robin/ml_loc_data',
        output_file_template = 'light_{}_{}.wav',
        base_dir = os.getcwd(),
        )

def simulate(args):
    import os, sys
    import numpy as np
    from scipy.io import wavfile

    signal = args[0]
    location = args[1]
    index = args[2]

    room_dim = parameters['room_dim']
    fs_sound = parameters['fs_sound']
    max_order = parameters['max_order']
    abosrption = parameters['absorption']
    sigma2_awgn = parameters['sigma2_awgn']
    mic_array = parameters['mic_array']
    fs_light = parameters['fs_light']
    output_dir = parameters['output_dir']
    base_dir = parameters['base_dir']
    output_file_template = parameters['output_file_template']

    sys.path.append(base_dir)
    from light_array import LightArrayMovingAverage

    # STEP 1 : Room simulation
    # Create the room
    room = pyroomacoustics.ShoeBox(room_dim,
            fs=fs_sound,
            max_order=max_order,
            absorption=abosrption,
            sigma2_awgn=sigma2_awgn)

    room.add_source(location)

    devices = LightArrayMovingAverage(np.array(mic_array).T, fs=fs_light)
    room.add_microphone_array(devices)

    outputs = dict()

    # add the source to the room
    G = 10 ** (signal['gain'] / 20)  # dB -> linear
    room.sources[0].signal = signal['data'] * G

    # Simulate sound transport
    room.simulate()

    # save the output
    filename = os.path.join(output_dir, output_file_template.format(signal['label'], index))
    wavfile.write(filename, fs_light, room.mic_array.signals.T)

    # add the filename to the dictionary
    metadata = signal.copy()
    metadata.pop('data')  # remove the data array
    metadata['location'] = location
    metadata['filename'] = filename

    return metadata

def generate_args(parameters):

    # Generate the training data on a grid
    # grid the room
    step = parameters['grid_step']  # place a source every 20 cm
    pos = [np.arange(step, L - step, step) for L in parameters['room_dim']]
    grid = itertools.product(*pos)

    # grid the gains
    gains = np.linspace(*parameters['gain_range'])

    # generate the argument for training data
    train_args = []
    index = 0
    for point in grid:
        for gain in gains:
            data = np.random.randn(int(parameters['fs_sound'] * parameters['sample_length']))
            train_args.append([ 
                dict(data=data, gain = gain, label='wn'),
                point,
                index,
                ])
            index += 1

    # Generate the test data at random locations in the room
    n_test = parameters['n_test']
    points = np.random.rand(n_test, 2) * np.array(parameters['room_dim'])[None,:]
    gains = np.random.uniform(low=parameters['gain_range'][0], high=parameters['gain_range'][1], size=n_test)

    test_args = []
    for point, gain in zip(points, gains):
        data = np.random.randn(int(parameters['fs_sound'] * parameters['sample_length']))
        test_args.append([ 
            dict(data=data, gain=gain.tolist(), label='wn',),
            tuple(point.tolist()),
            index,
            ])
        index += 1

    return train_args, test_args


if __name__ == '__main__':

    train_args, test_args = generate_args(parameters)

    c = ipp.Client()
    lbv = c.load_balanced_view()
    lbv.register_joblib_backend()
    print('{} engines waiting for orders.'.format(len(c[:])))

    toolbox = dict(
            simulate=simulate,
            parameters=parameters,
            )
    _ = c[:].push(toolbox, block=True)

    with c[:].sync_imports():
        import pyroomacoustics

    metadata_train = lbv.map_sync(simulate, train_args)
    metadata_test = lbv.map_sync(simulate, test_args)

    metadata = { 'train' : metadata_train, 'test' : metadata_test }

    filename = os.path.join(parameters['output_dir'], 'metadata_train_test.json')
    with open(filename, 'w') as f:
        json.dump(metadata, f)

