'''
This script generates simulated power intensity data to train
a localization network.
'''
import itertools, json, os
import datetime

import numpy as np
from scipy.io import wavfile

import matplotlib
matplotlib.use('agg')

import pyroomacoustics

import ipyparallel as ipp
from joblib import Parallel, delayed
from functools import partial

import jsongzip

parameters = dict(
        # simulation parameters
        sample_length = 0.5,      # length of audio sample in seconds
        gain_range = [-3., 3., 5],  # source gains, 5 points from -3dB to 3dB
        noise_var = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        grid_step = 0.05,
        n_test = 100,            # number of validation points
        n_validation = 1000,

        # room parameters
        room_dim = [6,5],
        fs_sound = 16000,
        max_order = 12,
        absorption = 0.2,

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

def split_speakers(seed=0, split=None):
    '''
    Split the speakers into train/validate/test

    Parameters
    ----------
    seed: int
        De-randomize by fixing the seed
    split: list of 3 int
        List containing the number of speakers to
        use in train/validate/test (in that order)
    '''
    import random

    if split is None:
        split = { 'train' : 2, 'validate' : 2, 'test' : 3 }

    speakers = pyroomacoustics.datasets.cmu_arctic_speakers

    # split male/female speakers to ensure good ratios
    spkr_f = list(filter(lambda s : speakers[s]['sex'] == 'female', speakers))
    spkr_m = list(filter(lambda s : speakers[s]['sex'] == 'male', speakers))

    random.seed(seed)
    random.shuffle(spkr_m)
    random.shuffle(spkr_f)

    pick = lambda L,n : [L.pop() for i in range(n)]

    sets = {
            'train' : pick(spkr_m, split['train']) + pick(spkr_f, split['train']),
            'validate' : pick(spkr_m, split['validate']) + pick(spkr_f, split['validate']),
            'test' : pick(spkr_m, split['test']) + pick(spkr_f, split['test']),
            }

    return sets

def simulate(args):
    import os, sys
    import numpy as np
    from scipy.io import wavfile

    signal = args[0]
    location = args[1]
    noise_var = args[2]
    index = args[3]

    room_dim = parameters['room_dim']
    fs_sound = parameters['fs_sound']
    max_order = parameters['max_order']
    abosrption = parameters['absorption']
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
            sigma2_awgn=noise_var)

    room.add_source(location)

    devices = LightArrayMovingAverage(np.array(mic_array).T, fs=fs_light)
    room.add_microphone_array(devices)

    outputs = dict()

    # add the source to the room
    G = 10 ** (signal['gain'] / 20)  # dB -> linear
    room.sources[0].signal = signal['data'] * G

    # Simulate sound transport
    room.simulate()

    # add the filename to the dictionary
    metadata = signal.copy()
    metadata.pop('data')  # remove the data array
    metadata['location'] = location
    metadata['filename'] = filename

    return [room.mic_array.signals.T.tolist(), np.r_[location, signal['gain'], noise_var].tolist()]

def filter_points(parameters, points, min_dist=0.1):
    '''
    Remove points that are too close to a microphone
    '''

    mic_array = np.array(parameters['mic_array'])

    dist = np.linalg.norm(points[:,None,:] - mic_array[None,:,:], axis=-1)
    I = np.all(dist > min_dist, axis=1)

    return points[I,:]

def generate_args(parameters, sound_type='wn'):

    # Generate the training data on a grid
    # grid the room
    step = parameters['grid_step']  # place a source every 20 cm
    pos = [np.arange(step, L - step, step) for L in parameters['room_dim']]
    grid = itertools.product(*pos)
    grid = filter_points(parameters, np.array(list(grid)))

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
                tuple(point.tolist()),
                0.,  # no noise
                index,
                ])
            index += 1

    # Generate the validation data at random locations in the room
    n_validation = parameters['n_validation']
    points = np.random.rand(n_validation, 2) * np.array(parameters['room_dim'])[None,:]
    points = filter_points(parameters, points)
    gains = np.random.uniform(low=parameters['gain_range'][0], high=parameters['gain_range'][1], size=n_validation)

    validation_args = []
    for point, gain in zip(points, gains):
        data = np.random.randn(int(parameters['fs_sound'] * parameters['sample_length']))
        validation_args.append([ 
            dict(data=data, gain=gain.tolist(), label='wn',),
            tuple(point.tolist()),
            0.,  # no noise
            index,
            ])
        index += 1

    # Generate the test data with various levels of noise
    n_test = parameters['n_test']
    points = np.random.rand(n_test, 2) * np.array(parameters['room_dim'])[None,:]
    points = filter_points(parameters, points)
    gains = np.random.uniform(low=parameters['gain_range'][0], high=parameters['gain_range'][1], size=n_test)

    test_args = []
    for point, gain in zip(points, gains):
        for var in parameters['noise_var']:
            data = np.random.randn(int(parameters['fs_sound'] * parameters['sample_length']))
            test_args.append([ 
                dict(data=data, gain=gain.tolist(), label='wn',),
                tuple(point.tolist()),
                var,
                index,
                ])
            index += 1

    return train_args, validation_args, test_args


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Create training data for ML-based localization')
    parser.add_argument('--perfect_model', action='store_true', help='Use a simple decay exponent rather than room simulation')
    parser.add_argument('--alpha', type=float, default=1., help='The decay exponent for the perfect model.')
    args = parser.parse_args()

    train_args, validate_args, test_args = generate_args(parameters, perfect_model=args.perfect_model, alpha=args.alpha)

    c = ipp.Client()
    lbv = c.load_balanced_view()
    lbv.register_joblib_backend()
    print('{} engines waiting for orders.'.format(len(c[:])))

    toolbox = dict(
            simulate=simulate,
            energy_decay=energy_decay,
            parameters=parameters,
            )
    _ = c[:].push(toolbox, block=True)

    with c[:].sync_imports():
        import matplotlib

    for engine in c:
        engine.apply(matplotlib.use, 'agg')

    with c[:].sync_imports():
        import pyroomacoustics

    if args.perfect_model:
        metadata_train = lbv.map_sync(energy_decay, train_args)
        metadata_validate = lbv.map_sync(energy_decay, validate_args)
        metadata_test = lbv.map_sync(energy_decay, test_args)
    else:
        metadata_train = lbv.map_sync(simulate, train_args)
        metadata_validate = lbv.map_sync(simulate, validate_args)
        metadata_test = lbv.map_sync(simulate, test_args)

    metadata = { 'parameters' : parameters, 'train' : metadata_train, 'validation' : metadata_validate, 'test' : metadata_test }

    now = datetime.datetime.now()
    timestamp = datetime.datetime.strftime(now, '%Y%m%d-%H%M%S')

    if args.perfect_model:
        filename = os.path.join(parameters['output_dir'], 'metadata_train_test_model_alpha{:.1f}.json'.format(args.alpha))
    else:
        filename = os.path.join(parameters['output_dir'], '{}_metadata_train_test.json'.format(timestamp))

    # save to gzipped json file
    jsongzip.dump(filename, metadata)

