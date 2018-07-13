import matplotlib
matplotlib.use('Agg')

import json, os
import numpy as np
import cupy as cp
import pandas as pd
import chainer
import seaborn as sns
import matplotlib.pyplot as plt

from ml_localization import get_data, models, get_formatters

def baseline(blinky_signals, blinky_locations, k_max=1):

    I = np.argsort(blinky_signals)[-k_max:]

    # create weights
    w = blinky_signals[I]
    w /= np.sum(w)

    # weighted combinations of blinky locations
    est = np.sum(w[:,None] * blinky_locations[I,:], axis=0)

    return est


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Run a model on a test set')
    parser.add_argument('protocol', type=str,
            help='The JSON file containing the experimental details.')
    parser.add_argument('config', type=str, help='The JSON file containing the configuration.')
    args = parser.parse_args()

    with open(args.protocol, 'r') as f:
        protocol = json.load(f)

    blinky_locations = np.array(protocol['blinky_locations'])

    with open(args.config, 'r') as f:
        config = json.load(f)

    # import model and use MSE
    nn = models[config['model']['name']](*config['model']['args'], **config['model']['kwargs'])
    chainer.serializers.load_npz(config['model']['file'], nn)

    # Helper to load the dataset
    if 'outputs' in config['data']['format_kwargs']:
        config['data']['format_kwargs'].pop('outputs')
    data_formatter, label_formatter, skip = get_formatters(outputs=(0,2), **config['data']['format_kwargs'])

    pickle_fn = '.pub_2018_ASJ_fall_test_{}.pickle'.format(config['name'])

    if not os.path.exists(pickle_fn):

        # Load the dataset
        train, validate, test = get_data(config['data']['file'],
                data_formatter=data_formatter, 
                label_formatter=label_formatter, skip=skip)

        table = []

        with chainer.using_config('train', False):

            for (example, label) in train:
                # get all samples with the correct noise variance
                e = np.squeeze(nn(example[:,:]).data) - label[:2]
                table.append([
                    'train',  # noise variance,
                    np.linalg.norm(e),
                    ] + e.tolist())

            for (example, label) in validate:
                # get all samples with the correct noise variance
                e = np.squeeze(nn(example[:,:]).data) - label[:2]
                table.append([
                    'validate',  # noise variance,
                    np.linalg.norm(e),
                    ] + e.tolist())

            for (example, label) in test:
                # get all samples with the correct noise variance
                e = np.squeeze(nn(example[:,:]).data) - label[:2]
                table.append([
                    'test',  # noise variance,
                    np.linalg.norm(e),
                    ] + e.tolist())

                # apply baseline on test data
                e_bl = baseline(example[0,:], blinky_locations) - label[:2]
                table.append([
                    'baseline $k=1$',  # noise variance,
                    np.linalg.norm(e_bl),
                    ] + e_bl.tolist())

                # apply baseline on test data
                e_bl = baseline(example[0,:], blinky_locations, k_max=4) - label[:2]
                table.append([
                    'baseline $k=4$',  # noise variance,
                    np.linalg.norm(e_bl),
                    ] + e_bl.tolist())

        df = pd.DataFrame(data=table, columns=['Set', 'Error', 'x', 'y'])

        df.to_pickle(pickle_fn)

    else:
        df = pd.read_pickle(pickle_fn)



    sns.violinplot(data=df, x='Set', y='Error')
    plt.ylim([0, 200])
    plt.savefig('pub_2018_ASJ_fall/mse_{}.pdf'.format(config['name']))

    # 90th percentile in L_inf norm (for test set)
    df_test = df[df['Set'] == 'test']
    p90 = np.percentile(np.maximum(np.abs(df_test['x']), np.abs(df_test['y'])), 90)
    blim = [-p90, p90]

    # first we find the 90 percentile (circularly)
    for set_ in ['train', 'validate', 'test', 'baseline $k=1$', 'baseline $k=4$']:
        sns.jointplot(x='x', y='y', data=df[df['Set'] == set_],
                kind='scatter', stat_func=None,
                xlim=blim, ylim=blim, s=2)
        plt.savefig('pub_2018_ASJ_fall/scatter_{}_{}.pdf'.format(config['name'], set_))

    plt.show()
