'''
This script generates simulated power intensity data to train
a localization network.
'''
import itertools, json, os, pickle
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

# cache file for the cmu corpus
cmu_arctic_cache_file = '.cmu_arctic_cache.pickle'
cmu_arctic = None  # this is a placeholder for the dataset
cmu_arctic_split = None  # this is a placeholder for the dataset

parameters = dict(
        # simulation parameters
        sample_length = 0.5,      # length of audio sample in seconds
        gain_range = [-3., 3., 5],  # source gains, 5 points from -3dB to 3dB
        noise_var = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        sound_types = ['white noise', 'CMU ARCTIC speech'],
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
        output_dir = '/datanet/robin/ml_loc_data',
        output_file_template = 'light_{}_{}.wav',
        base_dir = os.getcwd(),
        )


def load_cmu_arctic(basedir='/datanet/CMU_ARCTIC'):
    global cmu_arctic

    if os.path.exists(cmu_arctic_cache_file):
        # read from cache file
        with open(cmu_arctic_cache_file, 'rb') as f:
            cmu_arctic = pickle.load(f)
    else:
        # load the corpus
        cmu_arctic = pyroomacoustics.datasets.CMUArcticCorpus(basedir=basedir)
        # write cache file
        with open(cmu_arctic_cache_file, 'wb') as f:
            pickle.dump(cmu_arctic, f)

    return cmu_arctic


def cmu_arctic_splitter(cmu_arctic, seed=0, split=None):
    '''
    Split the speakers into train/validate/test

    Parameters
    ----------
    cmu_arctic: pyroomacoustics.datasets.CMUArcticCorpus
        The CMU ARCTIC corpus
    seed: int
        De-randomize by fixing the seed
    split: list of 3 int
        List containing the number of speakers to
        use in train/validate/test (in that order)
    '''
    import random

    if split is None:
        split = { 'train' : 0.4, 'validate' : 0.2, 'test' : 0.4 }


    # split male/female speakers to ensure good ratios
    speakers = pyroomacoustics.datasets.cmu_arctic_speakers
    spkr_f = list(filter(lambda s : speakers[s]['sex'] == 'female', speakers))
    spkr_m = list(filter(lambda s : speakers[s]['sex'] == 'male', speakers))

    # use only sentences existing for all speakers
    tags = []
    for tag, number in cmu_arctic.info['tag'].items():
        if number == len(speakers):
            tags.append(tag)
    n_tags = len(tags)

    # "randomize"
    random.seed(seed)
    random.shuffle(spkr_m)
    random.shuffle(spkr_f)
    random.shuffle(tags)

    # get the last few elements of a list
    pick = lambda L,n : [L.pop() for i in range(n)]

    sets = {}
    n_spkr = min(len(spkr_f), len(spkr_m))
    for subset in split.keys():
        sets[subset] = {}
        n_pick = round(n_spkr * split[subset])
        sets[subset]['speaker'] = pick(spkr_f, n_pick) + pick(spkr_m, n_pick)
        sets[subset]['tag'] = pick(tags, round(n_tags * split[subset]))



    sub_corpus = {}
    for subset, selectors in sets.items():
        sub_corpus[subset] = cmu_arctic.filter(**selectors)

    return sub_corpus

def cmu_arctic_random_sample(corpus, duration, normalize=True):
    ''' Returns a random sample of duration [s] from corpus '''

    # pick a sentence at random
    import random
    sentence = random.choice(corpus.samples)

    n_samples = int(duration * sentence.fs)
    trunc_len = (sentence.data.shape[0] // n_samples) * n_samples
    chopped = sentence.data[:trunc_len].reshape((-1,n_samples))

    # estimate power during silence
    silence_pwr = np.mean(sentence.data[:100]**2)

    # choose at random a frame with more than 2x power of silence
    chopped_pwr = np.mean(chopped.astype(np.float)**2, axis=1)
    I = np.where(chopped_pwr > 2 * silence_pwr)[0]
    frame_index = random.choice(I)
    rand_sample = chopped[random.choice(I)].astype(np.float)

    # normalize to have unit power
    if normalize:
        rand_sample /= np.linalg.norm(rand_sample)

    sample_id = '_'.join([sentence.meta.speaker, sentence.meta.tag, str(frame_index * n_samples)])


    return rand_sample, sample_id

    
def simulate(args):
    import os, sys
    import numpy as np
    from scipy.io import wavfile

    # extract arguments
    (sound_type, sid, signal, gain, location, noise_var, index) = args

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
    G = 10 ** (gain / 20)  # dB -> linear
    room.sources[0].signal = signal * G

    # Simulate sound transport
    room.simulate()

    labels = location + [noise_var, gain, label, sid, index]

    return [room.mic_array.signals.T.tolist(), np.r_[location, signal['gain'], noise_var].tolist()]

def filter_points(parameters, points, min_dist=0.1):
    '''
    Remove points that are too close to a microphone
    '''

    mic_array = np.array(parameters['mic_array'])

    dist = np.linalg.norm(points[:,None,:] - mic_array[None,:,:], axis=-1)
    I = np.all(dist > min_dist, axis=1)

    return points[I,:]

def get_data(sound_type, duration, fs, corpus=None):

    n_samples = int(duration * fs)

    if sound_type == 'white noise':
        data = np.random.randn(n_samples)
        sample_id = 'na'

    elif sound_type == 'CMU ARCTIC speech':
        if corpus is None:
            corpus = cmu_arctic
        data, sample_id = cmu_arctic_random_sample(corpus, duration)

    else:
        raise ValueError('Unsupported sound type: {}'.format(sound_type))

    return data, sample_id

def generate_args(parameters):

    # Generate the training data on a grid
    # grid the room
    step = parameters['grid_step']  # place a source every 20 cm
    pos = [np.arange(step, L - step, step) for L in parameters['room_dim']]
    grid = itertools.product(*pos)
    grid = filter_points(parameters, np.array(list(grid)))

    sound_types = parameters['sound_types']

    # grid the gains
    gains = np.linspace(*parameters['gain_range'])

    # The format of one line of args is
    ## sound_type, gain, point, noise_var, index

    # generate the argument for training data
    train_args = []
    index = 0
    for point, gain, snd_label in itertools.product(grid, gains, sound_types):
        sound, sid = get_data(snd_label, parameters['sample_length'], parameters['fs_sound'], corpus=cmu_arctic_split['train'])
        train_args.append([ snd_label, sid, sound, gain, tuple(point.tolist()), 0., index, ])
        index += 1

    # Generate the validation data at random locations in the room
    n_validation = parameters['n_validation']
    points = np.random.rand(n_validation, 2) * np.array(parameters['room_dim'])[None,:]
    points = filter_points(parameters, points)
    gains = np.random.uniform(low=parameters['gain_range'][0], high=parameters['gain_range'][1], size=n_validation)

    validation_args = []
    for point, gain in zip(points, gains):
        for snd_label in sound_types:
            sound, sid = get_data(snd_label, parameters['sample_length'], parameters['fs_sound'], corpus=cmu_arctic_split['validate'])
            validation_args.append([ snd_label, sid, sound, gain, tuple(point.tolist()), 0., index, ])
            index += 1

    # Generate the test data with various levels of noise
    n_test = parameters['n_test']
    points = np.random.rand(n_test, 2) * np.array(parameters['room_dim'])[None,:]
    points = filter_points(parameters, points)
    gains = np.random.uniform(low=parameters['gain_range'][0], high=parameters['gain_range'][1], size=n_test)

    test_args = []
    for point, gain in zip(points, gains):
        for snd_label in sound_types:
            for var in parameters['noise_var']:
                sound, sid = get_data(snd_label, parameters['sample_length'], parameters['fs_sound'], corpus=cmu_arctic_split['test'])
                test_args.append([ snd_label, sid, sound, gain, tuple(point.tolist()), var, index, ])
                index += 1

    return train_args, validation_args, test_args


if __name__ == '__main__':
    global cmu_subsets

    import argparse
    parser = argparse.ArgumentParser(description='Create training data for ML-based localization')
    args = parser.parse_args()

    # Load the database
    cmu_arctic = load_cmu_arctic()
    cmu_arctic_split = cmu_arctic_splitter(cmu_arctic)

    # Generate the arguments
    train_args, validate_args, test_args = generate_args(parameters)

    import pdb
    pdb.set_trace()

    # Start parallel stuff
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

    # Now run the parallel beast
    metadata_train = lbv.map_sync(simulate, train_args)
    metadata_validate = lbv.map_sync(simulate, validate_args)
    metadata_test = lbv.map_sync(simulate, test_args)

    metadata = { 'parameters' : parameters, 'train' : metadata_train, 'validation' : metadata_validate, 'test' : metadata_test }

    now = datetime.datetime.now()
    timestamp = datetime.datetime.strftime(now, '%Y%m%d-%H%M%S')

    filename = os.path.join(parameters['output_dir'], '{}_metadata_train_test.json'.format(timestamp))

    # save to gzipped json file
    jsongzip.dump(filename, metadata)

