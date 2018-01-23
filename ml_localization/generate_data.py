'''
This script generates simulated power intensity data to train
a localization network.
'''
import itertools, json

import numpy as np
from scipy.io import wavfile

import pyroomacoustics as pra

import ipyparallel as ipp
from joblib import Parallel, delayed
from functools import partial

parameters = dict(
        # simulation parameters
        n_epochs = 1,
        sample_length = 3.,      # length of audio sample in seconds
        gain_range = [-3., 3., 5],  # source gains, 5 points from -3dB to 3dB
        grid_step = 0.05,

        # room parameters
        room_dim = [6,5],
        fs_sound = 16000,
        max_order = 12,
        abosrption = 0.4,
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
        fs_light = 30,

        # output parameters
        output_dir = './data',
        output_file_template = 'light_{label}_{gain}_{index}.wav'
        )

def generate_data(signals, location, index,
        room_dim=(6,5), fs_sound=16000, 
        max_order=10, abosrption=0.4, sigma2_awgn=0., 
        mic_array=None, fs_light=30,
        output_dir='./data'):
    '''
    Parameters
    ----------
    signals: dict
        A dictionary with each entry itself a dictionary with entries 'data' and 'gains'
    location: array_like
        The 2D location of the source
    index: int
        A unique identifier for this signal
    room_dim: array_like
        The size of the ShoeBox room
    fs_sound: int
        The sampling frequency of the sound signals
    max_order: int
        The maximum order of image sources to use
    abosrption: float
        The aborption of the walls in the room
    sigma2_awgn: float
        The variance of additive gaussian white noise
    mic_array: array_like
        The coordinates of the microphones in a (n_dim x n_mics) shaped array
    fs_light: int
        The sampling frequency of the light signal
    '''

    # STEP 1 : Room simulation
    # Create the room
    room = ShoeBox(room_dim,
            fs=fs_sound,
            max_order=max_order,
            absorption=abosrption,
            sigma2_awgn=sigma2_awgn)

    room.add_source(location, signal=G * signal)

    devices = LightArray(device_locations, fs=fs_light)
    room.add_microphone_array(devices)

    outputs = dict()

    for label in signals.keys():
        # add the source to the room
        signal = signals[label].pop('data')  # remove data from dictionary
        G = 10 ** (signals[label]['gain'] / 20)  # dB -> linear
        room.sources[0].signal = signal * G

        # Simulate sound transport
        room.simulate()

        # save the output
        filename = os.path.join(output_dir, output_file_template.format(label, gain, index))
        wavfile.write(filename, fs_light, room.mic_array.signals.T)

        # add the filename to the dictionary
        signals[label]['location'] = location.tolist()
        signals[label]['filename'] = filename

    return signals

if __name__ == '__main__':


    # grid the room
    step = parameters['grid_step']  # place a source every 20 cm
    pos = [np.arange(step, L - step, step) for L in parameters['room_dim']]
    grid = itertools.product(*pos)

    # grid the gains
    gains = np.linspace(*parameters['gain_range'])

    # generate the arguments
    args = []
    for index, point in enumerate(grid):
        for gain in gains:
            data = np.random.randn(int(parameters['fs_sound'] * parameters['sample_length']))
            args.append([ 
                { 'wn' : dict(data=data, gain = gain,), },
                point,
                index,
                ])

    c = ipp.Client()
    lbv = c.load_balanced_view()
    lbv.register_joblib_backend()
    print('{} engines waiting for orders.'.format(len(c[:])))

    func = partial(mapping_error, **parameters)
    metadata = Parallel(backend='ipyparallel')(delayed(func)(*arg) for arg in args)

    filename = os.path.join(parameters['output_dir'], 'metadata.json')
    with open(filename, 'w') as f:
        json.dump(metadata, f)

