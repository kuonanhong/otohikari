import os, json
import numpy as np
import cupy as cu
from scipy.io import wavfile

from chainer.datasets import TupleDataset

def read_data(data_list):

    dataset = []
    min_len = np.inf
    i = 0
    for sample in data_list:

        _, data = wavfile.read(sample['filename'])
        avg = data[-16:-1,:].mean(axis=0, keepdims=True)
        
        if np.any(np.isnan(avg)):
            print('Skip a sample with nan')
            continue

        dataset.append(
                # It is important that this is a *tuple*!
                (cu.array(avg.tolist(), dtype=np.float32), cu.array(sample['location'], dtype=np.float32),)
                )

    return dataset

def get_loc_data(metadata_file):

    with open(metadata_file, 'r') as f:
        data_list = json.load(f)
    
    train = read_data(data_list['train'])
    test = read_data(data_list['test'])

    return train, test
