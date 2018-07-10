import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import pandas as pd
import chainer
import seaborn as sns
import matplotlib.pyplot as plt

from ml_localization import get_data, models, get_formatters

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Run a model on a test set')
    parser.add_argument('config', type=str, help='The JSON file containing the configuration.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # import model and use MSE
    nn = models[config['model']['name']](*config['model']['args'], **config['model']['kwargs'])
    chainer.serializers.load_npz(config['model']['file'], nn)

    # Helper to load the dataset
    if 'outputs' in config['data']['format_kwargs']:
        config['data']['format_kwargs'].pop('outputs')
    data_formatter, label_formatter, skip = get_formatters(outputs=(0,4), **config['data']['format_kwargs'])

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

    df = pd.DataFrame(data=table, columns=['Set', 'Error', 'x', 'y'])

    sns.violinplot(data=df, x='Set', y='Error')
    plt.ylim([0, 200])
    plt.savefig('pub_2018_ASJ_fall/mse_{}.pdf'.format(config['name']))

    plt.figure()
    sns.jointplot(x='x', y='x', data=df[df['Set'] == 'test'])
    plt.savefig('pub_2018_ASJ_fall/scatter_{}.pdf'.format(config['name']))
    plt.show()
