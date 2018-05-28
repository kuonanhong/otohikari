from __future__ import division, print_function

import sys, argparse, copy, os, re
import numpy as np
import pandas as pd
import json

if 'DISPLAY' not in os.environ:
    import matplotlib as mpl
    mpl.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
            description='Plot the data simulated by separake_near_wall')
    parser.add_argument('-p', '--pickle', action='store_true', 
            help='Read the aggregated data table from a pickle cache')
    parser.add_argument('-s', '--show', action='store_true',
            help='Display the plots at the end of data analysis')
    parser.add_argument('dirs', type=str, nargs='+', metavar='DIR',
            help='The directory containing the simulation output files.')

    cli_args = parser.parse_args()
    plot_flag = cli_args.show
    pickle_flag = cli_args.pickle

    # metrics to take in the plot
    columns = ['id','error']

    parameters = dict()
    args = []
    df = None

    for i, data_dir in enumerate(cli_args.dirs):

        print('Reading in', data_dir)

        # Read in the parameters
        with open(data_dir + '/parameters.json', 'r') as f:
            parameters_part = json.load(f)

            for key, val in parameters_part.items():
                if key in parameters and val != parameters[key]:
                    print('Warning ({}): {}={} (vs {} in {})'.format(data_dir, 
                        key, val, parameters[key], cli_args.dirs[0]))

            else:
                if i > 0:
                    print('Warning ({}): parameter {}={} was not present before'.format(data_dir, key, val))
                parameters[key] = val

        with open(data_dir + '/arguments.json', 'r') as f:
            args.append(json.load(f))

        # check if a pickle file exists for these files
        pickle_file = data_dir + '/dataframe.pickle'

        if os.path.isfile(pickle_file) and pickle_flag:
            print('Reading existing pickle file...')
            # read the pickle file
            df_part = pd.read_pickle(pickle_file)

        else:

            # reading all data files in the directory
            records = []
            for file in os.listdir(data_dir):
                if file.startswith('data') and file.endswith('.json'):
                    with open(os.path.join(data_dir, file), 'r') as f:
                        records += json.load(f)

            # build the data table line by line
            print('  Building table')
            table = []

            src_ind_2_label = {0: 'Female', 1:'Male'}
            for record in records:
                gt_locs = np.array(record['groundtruth_locations'])
                recon_locs = np.array(record['reconstructed_locations'])

                for i,(gt,recon) in enumerate(zip(gt_locs, recon_locs)):
                    table.append([i, np.linalg.norm(gt - recon)])
               
            # create a pandas frame
            print('  Making PANDAS frame...')
            df_part = pd.DataFrame(table, columns=columns)
            df_part.to_pickle(pickle_file)

        if df is None:
            df = df_part
        else:
            df = pd.concat([df, df_part], ignore_index=True)

    # Draw the figure
    print('Plotting...')

    # get a basename for the plot
    plot_basename = ('figures/' + cli_args.dirs[0].rstrip('/').split('/')[-1])

    sns.set(style='whitegrid', context='paper', 
            #palette=sns.light_palette('navy', n_colors=2),
            #palette=sns.light_palette((210, 90, 60), input="husl", n_colors=2),
            font_scale=0.9,
            rc={
                'figure.figsize':(3.38649 * 0.8, 3.338649 / 3), 
                'lines.linewidth':1.,
                #'font.family': u'Roboto',
                #'font.sans-serif': [u'Roboto Bold'],
                'text.usetex': False,
                })

    error = df['error']
    p25, p50, p75, p90, p95 = np.percentile(error, [25, 50, 75, 90, 95])
    max_right = 0.5
    n_buckets = 50
    heights, buckets = np.histogram(error, bins=np.linspace(0, max_right, n_buckets))
    w = buckets[1] - buckets[0]
    Z = len(error)
    plt.bar(buckets[:-1], heights / Z, width=w)

    p_cut = (1 - np.sum(heights)/Z)
    print('{:.2f}% cut on the right'.format(p_cut * 100))
    
    # paint red the center 50%
    #fifty = np.logical_and(buckets[:-1] > p25, buckets[:-1] < p75)
    #plt.bar(buckets[:-1][fifty], heights[fifty] / Z, width=w, color='r')

    # indicate median and p_90
    for p, note in zip([p50, p90], ['50', '90']):

        # find closest bucket
        cl = np.argmin(np.abs(buckets - p))
        if p - buckets[cl] < 0.:
            cl -= 1

        h = heights[cl] / Z

        plt.annotate('$p_{{{}}}$'.format(note), xy=[p, h], xytext=[p+0.02, h+0.02],
                va="bottom", ha="center",
                arrowprops=dict(arrowstyle="->", lw=1., facecolor='black'),
                fontsize='small', 
                annotation_clip=False)

    plt.annotate('${:.2f}\%$ cut-off'.format(100*p_cut), xy=[max_right, 0.01], xytext=[max_right-0.05, 0.01],
        va="center", ha="right",
        arrowprops=dict(arrowstyle="->", lw=1., facecolor='black'),
        fontsize='small', 
        annotation_clip=False)

    plt.yticks([])
    plt.xlabel('Localization Error [m]')
    sns.despine(left=True)

    plt.tight_layout(pad=0.05)

    plt.savefig(plot_basename + '_hist.pdf', dpi=300)

    if plot_flag:
        plt.show()

