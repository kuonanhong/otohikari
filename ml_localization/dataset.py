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

def get_loc_perfect_model_data(metadata_file):

    with open(metadata_file, 'r') as f:
        data_list = json.load(f)

    train = data_list['train']
    test = data_list['test']

    def format_examples(examples):
        formated_examples = []
        for example in examples:
            example[0] = cu.array(example[0], dtype=np.float32)
            example[1] = cu.array(example[1][:-1], dtype=np.float32)

            if not (cu.all(cu.isfinite(example[0])) and cu.all(cu.isfinite(example[1]))) or cu.any(example[0] > 1000.):
                print('skip nan')
                continue

            formated_examples.append(tuple(example))

        return formated_examples

    def check_examples(examples):

        clean = True

        for example in examples:
            if cu.any(example[0] > 1000.):
                print('oups big:', example[0])
                clean = False

        if not clean:
            print('Some problematic examples')
        else:
            print('All examples are ok')


    
    train = format_examples(train)
    test = format_examples(test)

    check_examples(train)
    check_examples(test)

    return train, test
