import os, json
import numpy as np
from scipy.io import wavfile

def get_data(metadata_file, data_formatter=None, label_formatter=None, skip=None):

    with open(metadata_file, 'r') as f:
        data_list = json.load(f)

    train = data_list['train']
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
    test = format_examples(test)

    return train, test
