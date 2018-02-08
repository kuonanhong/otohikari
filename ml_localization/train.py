#!/usr/bin/env python
from __future__ import print_function

import argparse, os

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

# Get the data for training
from ml_localization import get_data
data_folder = '/data/robin/ml_loc_data'
metadata_fn = os.path.join(data_folder, 'metadata_train_test.json')
metadata_perfmodel_fn = os.path.join(data_folder, 'metadata_train_test_test_model_alpha_1.0.json')


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units, n_units)  # n_units -> n_units
            self.l3 = L.Linear(n_units, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    parser.add_argument('--perfect_model', action='store_true', help='Use the perfect model data.')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    devices = {'main':0, 'second':1, 'third':2, 'fourth':3}
    chainer.cuda.get_device_from_id(devices['main']).use()

    # Set up a neural network to train
    # Classifier reports mean squared error
    nn = MLP(args.unit, 2)
    model = L.Classifier(nn, lossfun=F.mean_squared_error)
    model.compute_accuracy=False

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    #optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Helper to load the dataset
    if args.perfect_model:
        def data_formatter(e):
            return np.array(e, dtype=np.float32)[None,:]
    else:
        def data_formatter(e):
            #return np.array(e, dtype=np.float32)[1:16,:].mean(axis=0, keepdims=True)
            #return np.array(e, dtype=np.float32)[1:16,:].reshape((1,-1))
            return np.array(e, dtype=np.float32)[1:5,:].mean(axis=0, keepdims=True)

    def label_formatter(l):
        return np.array(l[:2], dtype=np.float32)

    def skip (e):
        all_finite = np.all(np.isfinite(e[0])) and np.all(np.isfinite(e[1]))
        return not (all_finite and np.all(e[0] < 1000.))

    fn = metadata_perfmodel_fn if args.perfect_model else metadata_fn

    # Load the dataset
    train, test = get_data(fn,
            data_formatter=data_formatter, 
            label_formatter=label_formatter, skip=skip)

    print('Type of first example:', type(train[0][0]))
    wrong_type = False
    for example in train:
        if type(example[0]) != type(train[0][0]):
            wrong_type = True
    if wrong_type:
        print('Detected some type inconsistensies')

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.ParallelUpdater(train_iter, optimizer, devices=devices)
    '''
    updater = training.StandardUpdater(
        train_iter, optimizer, device=0)
    '''
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=devices['main']))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    return nn, train, test



if __name__ == '__main__':
    nn, train, test = main()
