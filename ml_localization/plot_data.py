import os, json
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma

# Get the data for training
from ml_localization import get_data
data_folder = '/Users/scheibler/Documents/Research/TMU/LightSound/code'
metadata_fn = os.path.join(data_folder, 'metadata_train_test.json')
metadata_perfmodel_fn = os.path.join(data_folder, 'metadata_train_test_test_model_alpha_1.0.json')

parameters = dict(
        # simulation parameters
        sample_length = 0.5,      # length of audio sample in seconds
        gain_range = [-3., 3., 5],  # source gains, 5 points from -3dB to 3dB
        grid_step = 0.05,
        n_test = 1000,

        # room parameters
        room_dim = [6,5],
        fs_sound = 16000,
        max_order = 12,
        absorption = 0.4,
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
        fs_light = 32,

        # output parameters
        output_dir = '/data/robin/ml_loc_data',
        output_file_template = 'light_{}_{}.wav',
        base_dir = os.getcwd(),
        )


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Inspect the training data')
    parser.add_argument('--perfect_model', action='store_true', help='Use perfect model training data')
    args = parser.parse_args()

    # Helper to load the dataset
    if args.perfect_model:
        def data_formatter(e):
            return np.array(e, dtype=np.float32)[None,:]
    else:
        def data_formatter(e):
            return np.array(e, dtype=np.float32)[1:16,:].mean(axis=1, keepdims=True)

    def label_formatter(l):
        return np.array(l, dtype=np.float32)[None,:]

    def skip (e):
        all_finite = np.all(np.isfinite(e[0])) and np.all(np.isfinite(e[1]))
        select_gain = e[1][0,2] == 0.
        return not (all_finite and np.all(e[0] < 1000.)) and select_gain

    fn = metadata_perfmodel_fn if args.perfect_model else metadata_fn

    train, test = get_data(fn,
            data_formatter=data_formatter, 
            label_formatter=label_formatter, skip=skip)

    # extract from training data
    cache_file = 'simulated_training_data.npz'
    if not os.path.exists(cache_file):
        train, test = get_loc_data(metadata_fn)
        power = np.squeeze(np.array([t[0] for t in train]))
        points = np.array([t[1][:2] for t in train])
        np.savez(cache_file, power=power, points=points)
    else:
        data = np.load(cache_file)
        points = data['points']
        power = data['power']

    # normalize
    power -= power.min()
    power /= power.max()

    # Generate the training data on a grid
    # grid the room
    step = parameters['grid_step']  # place a source every 20 cm
    xi, yi = [np.arange(step, L - step, step) for L in parameters['room_dim']]


    for i,p in enumerate(power.T):

        plt.subplot(4,2,i+1)

        '''
        # grid the data.
        zi = griddata(points, p, (xi[None,:], yi[:,None]), method='cubic')

        #plt.scatter(points[:,0], points[:,1], 
        th_clip = zi.min() + (zi.max() - zi.min()) / 2.
        zi[np.where(zi > th_clip)] = th_clip 

        # contour the gridded data, plotting dots at the randomly spaced data points.
        #CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
        CS = plt.pcolormesh(xi,yi,zi)
        plt.colorbar() # draw colorbar
        '''
        # plot data points.
        cmap = plt.get_cmap('viridis')
        colors = [cmap(pw) for pw in p]
        plt.scatter(points[:,0], points[:,1], p, c=colors)
        plt.xlim([0, parameters['room_dim'][0]])
        plt.ylim([0, parameters['room_dim'][1]])
        plt.title('{}th sensor'.format(i))

    plt.show()
