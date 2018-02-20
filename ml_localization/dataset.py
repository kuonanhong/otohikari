import os, json
import jsongzip
import numpy as np
from scipy.io import wavfile

def get_formatters(method='reshape', frames=(1,16), outputs=(0,2)):
    '''
    Return the data formating function

    Parameters
    ----------
    method: str
        Either 'reshape' or 'avg'
    frames: int or slice
        The frames to use
    outputs: int or slice
        The outputs to use
    '''

    if method == 'reshape':

        def data_formatter(e):
            return np.array(e, dtype=np.float32)[slice(*frames),:].reshape((1,-1))

    elif method == 'avg':

        def data_formatter(e):
            return np.array(e, dtype=np.float32)[slice(*frames),:].mean(axis=0, keepdims=True)

    def label_formatter(l):
        return np.array(l[slice(*outputs)], dtype=np.float32)

    def skip(e):
        all_finite = np.all(np.isfinite(e[0])) and np.all(np.isfinite(e[1]))
        return not (all_finite and np.all(e[0] < 1000.))

    return data_formatter, label_formatter, skip


def get_data(metadata_file, data_formatter=None, label_formatter=None, skip=None):

    data_list = jsongzip.load(metadata_file)

    train = data_list['train']
    validation = data_list['validation']
    test = data_list['test']

    def format_examples(examples):
        formated_examples = []
        for example in examples:

            if data_formatter is not None:
                example[0] = data_formatter(example[0])

            if label_formatter is not None:
                example[1] = label_formatter(example[1])

            if skip(example):
                print('skip nan')
                continue

            formated_examples.append(tuple(example))

        return formated_examples

    train = format_examples(train)
    validation = format_examples(validation)
    test = format_examples(test)

    return train, validation, test
